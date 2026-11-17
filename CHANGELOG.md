# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- SVM no longer corrects class imbalance twice (undersampling plus class weights)

### Added
- Validation-based hyperparameter search (`grid` in the model config), used for SVM C and gamma

### Added
- Data validation, time-based split and shared preprocessing pipeline
- Logistic regression, calibrated RBF SVM and XGBoost models
- Cost-based review threshold and precision-targeted block threshold
- Versioned model registry with reproducibility metadata
- Test suite on synthetic data, Ruff linting and GitHub Actions CI
- Architecture document and ADRs 0001 to 0004
