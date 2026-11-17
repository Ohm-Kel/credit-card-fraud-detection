# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
