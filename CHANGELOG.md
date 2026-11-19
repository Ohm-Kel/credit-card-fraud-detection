# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Keras neural network (`keras_mlp`) as a scikit-learn estimator with dropout and early stopping
- Optional `dl` extra for TensorFlow and a CI job that tests it
- ADR 0006
- Keras results in the README
- SHAP explanations (`fraud-explain`): global importance chart and top reasons per alert
- Optional `explain` extra and ADR 0007
- SHAP findings for XGBoost in the README
- FastAPI scoring service (`/health`, `/predict`, `/predict/batch`) with decisions, thresholds and SHAP reasons
- Dockerfile, Docker Compose and a CI job that builds the image and scores an example transaction
- Synthetic demo model script and example request; ADR 0008
- Prediction logging in the API (`FRAUD_PREDICTION_LOG`), non-fatal on write errors
- `fraud-monitor`: PSI drift on features and scores, PR-AUC check with delayed labels, exit code 1 on alerts; ADR 0009
- Optional MLflow tracking (`fraud-train --track`, `tracking` extra): parent run per training, nested run per model; ADR 0010

## [0.1.0] - 2026-11-17

### Added
- Data validation, time-based split and shared preprocessing pipeline
- Logistic regression, calibrated RBF SVM and XGBoost models
- Validation-based hyperparameter search (`grid` in the model config), used for SVM C and gamma
- Cost-based review threshold and precision-targeted block threshold
- Versioned model registry with reproducibility metadata
- Test suite on synthetic data, Ruff linting and GitHub Actions CI
- Architecture document and ADRs 0001 to 0005
- First evaluation results on the Kaggle dataset

### Fixed
- SVM no longer corrects class imbalance twice (undersampling plus class weights)

[Unreleased]: https://github.com/Ohm-Kel/credit-card-fraud-detection/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ohm-Kel/credit-card-fraud-detection/releases/tag/v0.1.0
