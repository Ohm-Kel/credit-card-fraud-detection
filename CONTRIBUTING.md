# Contributing

## Workflow

1. Branch off `main`: `feat/<name>`, `fix/<name>`, `docs/<name>` or `chore/<name>`.
2. Commit with [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary`.
3. Open a pull request describing what changed, why, and how it was tested.
4. CI (lint and tests) must pass before merging.

## Local checks

```bash
pip install -e ".[dev]"
pre-commit install
ruff check . && ruff format --check .
pytest
```

## Modelling changes

Any change to features, models or thresholds must include updated metrics in the pull request
(validation and test PR-AUC, recall, precision and expected cost), produced with the fixed seed in
`configs/train.yaml`. Record significant design decisions as an ADR in `docs/adr/`.
