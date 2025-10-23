# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-23

### Added
- Initial release
- Core `select_optimizer()` function with decision tree logic
- `get_sklearn_model()` helper for sklearn integration
- Support for L1, L2, elastic net, and no regularization
- Memory constraint checking
- Comprehensive documentation:
  - Quick reference guide
  - Theoretical background on optimization trade-offs
  - Multiple example scripts
- Type hints throughout codebase
- GPL v3 License
- Setup.py for package installation
- Contributing guidelines

### Features
- Automatic solver selection based on:
  - Number of parameters
  - Number of samples
  - Regularization type
  - Memory constraints
  - Whether dataset fits in memory
- Support for classification and regression tasks
- sklearn-compatible API
- Detailed explanations for each recommendation

### Documentation
- README with installation instructions and examples
- Quick reference guide with common scenarios
- Theoretical background on optimization methods
- Lab-specific examples
- Basic demonstration scripts

### Examples
- `quickstart.py` - Simple getting started example
- `optimizer_demo.py` - Comprehensive demonstrations
- `use_with_labs.py` - Integration with lab assignments

## [Unreleased]

### Planned
- Support for PyTorch/TensorFlow optimizers
- Automatic hyperparameter tuning integration
- Benchmarking suite for different problem types
- Visualization tools for decision tree
- Neural network optimizer support
- More sophisticated memory estimation

