# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-19

### Added
- Keras neural network (`keras_mlp`) as a scikit-learn estimator with dropout and early stopping (`dl` extra, ADR 0006)
- SHAP explanations with `fraud-explain`: global importance chart and top reasons per alert (`explain` extra, ADR 0007)
- FastAPI scoring service (`/health`, `/predict`, `/predict/batch`) returning the decision, thresholds, model version and SHAP reasons (`serve` extra, ADR 0008)
- Docker image, Docker Compose, and a CI job that builds the image and scores an example transaction
- Synthetic demo model script and example request, so the API runs without the Kaggle file
- Prediction logging in the API and `fraud-monitor` for PSI drift and delayed-label performance checks (ADR 0009)
- Optional MLflow tracking with `fraud-train --track` (`tracking` extra, ADR 0010)
- Keras results and SHAP findings in the README

### Changed
- SVM search grid widened to C in {1, 10, 100} and gamma in {0.01, 0.001, 0.0001}; the previous best setting sat at the grid edge
- Synthetic data generator moved from the tests into the package for reuse by CI and demos

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

[Unreleased]: https://github.com/Ohm-Kel/credit-card-fraud-detection/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Ohm-Kel/credit-card-fraud-detection/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ohm-Kel/credit-card-fraud-detection/releases/tag/v0.1.0
