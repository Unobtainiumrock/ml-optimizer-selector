# Contributing to ML Optimizer Selector

Thank you for your interest in contributing to this project! This utility was created for educational purposes to help machine learning practitioners select the optimal solver for their problems.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the issue tracker
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, scikit-learn version)

### Suggesting Enhancements

We welcome suggestions for:

- Additional solvers or optimization methods
- Better heuristics for solver selection
- Support for other ML algorithms (SVM, neural networks, etc.)
- Improved memory estimation
- Better documentation or examples

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure code follows the existing style:
   - Type hints for all functions
   - Comprehensive docstrings (NumPy style)
   - Clean, readable code
5. Test your changes with the demo scripts
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions (NumPy style)
- Keep functions focused and single-purpose
- Add examples in docstrings where helpful

### Testing

Before submitting a PR:

```bash
# Run the demos to ensure nothing broke
python optimizer_selector.py
python optimizer_demo.py
python use_with_labs.py
```

### Areas for Improvement

Some ideas for contributions:

1. **Better memory estimation**: Current L-BFGS memory calculation is approximate
2. **Solver benchmarking**: Empirical comparison across different problem types
3. **Neural network support**: Extend to PyTorch/TensorFlow optimizers
4. **Auto-tuning**: Automatically select regularization strength
5. **Visualization**: Plot decision boundaries or convergence curves
6. **More examples**: Real-world datasets and use cases

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

