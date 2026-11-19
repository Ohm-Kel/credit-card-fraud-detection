import pytest

pytest.importorskip("fastapi")
pytest.importorskip("shap")

from fastapi.testclient import TestClient  # noqa: E402

from fraud_detection.data.split import time_based_split  # noqa: E402
from fraud_detection.features.build import feature_columns  # noqa: E402
from fraud_detection.models.factory import fit_model  # noqa: E402
from fraud_detection.models.registry import save_model  # noqa: E402
from fraud_detection.serving.app import create_app  # noqa: E402

COLS = feature_columns()
POLICY = {"review_threshold": 0.3, "block_threshold": 0.9}


def _save(tmp_path, transactions, name, params, policy=POLICY):
    split = time_based_split(transactions)
    model = fit_model(name, params, split.train[COLS], split.train["Class"], seed=0)
    folder = save_model(model, name, {"policy": policy}, tmp_path / "models")
    return folder, split


@pytest.fixture
def xgb_client(tmp_path, transactions):
    folder, split = _save(tmp_path, transactions, "xgboost", {"n_estimators": 30, "max_depth": 3})
    with TestClient(create_app(folder)) as client:
        yield client, split


def _payload(row):
    return {col: float(row[col]) for col in COLS}


def test_health(xgb_client):
    client, _ = xgb_client
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_name"] == "xgboost"
    assert body["explanations_available"] is True


def test_predict_returns_consistent_decision_and_reasons(xgb_client):
    client, split = xgb_client
    for _, row in split.test.head(20).iterrows():
        body = client.post("/predict", json=_payload(row)).json()
        p = body["fraud_probability"]
        expected = "block" if p >= 0.9 else "review" if p >= 0.3 else "approve"
        assert body["decision"] == expected
        assert len(body["reasons"]) == 3
        contributions = [r["contribution"] for r in body["reasons"]]
        assert contributions == sorted(contributions, reverse=True)


def test_fraud_scores_higher_than_legit(xgb_client):
    client, split = xgb_client
    test = split.test
    fraud = client.post("/predict", json=_payload(test[test["Class"] == 1].iloc[0])).json()
    legit = client.post("/predict", json=_payload(test[test["Class"] == 0].iloc[0])).json()
    assert fraud["fraud_probability"] > legit["fraud_probability"]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.pop("V5"),
        lambda p: p.update(Amount=-1.0),
        lambda p: p.update(Time=-10.0),
        lambda p: p.update(Class=1),
        lambda p: p.update(V1="not a number"),
    ],
    ids=["missing-field", "negative-amount", "negative-time", "extra-field", "wrong-type"],
)
def test_invalid_input_rejected(xgb_client, change):
    client, split = xgb_client
    payload = _payload(split.test.iloc[0])
    change(payload)
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_matches_single(xgb_client):
    client, split = xgb_client
    rows = [_payload(r) for _, r in split.test.head(5).iterrows()]
    batch = client.post("/predict/batch", json={"transactions": rows}).json()["predictions"]
    singles = [client.post("/predict", json=r).json() for r in rows]
    assert [b["fraud_probability"] for b in batch] == pytest.approx(
        [s["fraud_probability"] for s in singles]
    )


def test_empty_batch_rejected(xgb_client):
    client, _ = xgb_client
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_non_tree_model_serves_without_reasons(tmp_path, transactions):
    folder, split = _save(
        tmp_path,
        transactions,
        "logistic_regression",
        {"max_iter": 500},
        policy={"review_threshold": 0.5, "block_threshold": None},
    )
    with TestClient(create_app(folder)) as client:
        assert client.get("/health").json()["explanations_available"] is False
        body = client.post("/predict", json=_payload(split.test.iloc[0])).json()
        assert body["reasons"] is None
        assert body["block_threshold"] is None
        assert body["decision"] in {"approve", "review"}


def test_openapi_docs_available(xgb_client):
    client, _ = xgb_client
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]
