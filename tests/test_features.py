import numpy as np

from fraud_detection.features.build import (
    build_preprocessor,
    encode_time_of_day,
    feature_columns,
)


def test_time_encoding_is_cyclical():
    midnight, almost_midnight = encode_time_of_day(np.array([0.0, 86_399.0]))
    assert np.allclose(midnight, almost_midnight, atol=1e-3)
    next_day = encode_time_of_day(np.array([86_400.0 + 3_600]))
    one_am = encode_time_of_day(np.array([3_600.0]))
    assert np.allclose(next_day, one_am)


def test_preprocessor_output_shape_and_names(transactions):
    preprocessor = build_preprocessor()
    output = preprocessor.fit_transform(transactions[feature_columns()])
    assert output.shape == (len(transactions), 2 + 1 + 28)
    names = list(preprocessor.get_feature_names_out())
    assert names[:3] == ["time_of_day_sin", "time_of_day_cos", "Amount"]
    assert not np.isnan(output).any()


def test_target_is_not_a_feature():
    assert "Class" not in feature_columns()
