# Contributing to PSO Portfolio Optimizer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/PSO_optimal.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest-cov

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document functions with docstrings
- Format code with Black: `black src/ app/ tests/`
- Check linting: `flake8 src/ app/ tests/`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pso.py -v
```

## Documentation

- Update docstrings for new functions/classes
- Update relevant markdown files in `docs/`
- Add examples for new features
- Update README.md if needed

## Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: What changes and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs as needed
- **Code Quality**: Pass all tests and linting

## Areas for Contribution

### High Priority

- [ ] Add more optimization algorithms (Genetic Algorithm, Differential Evolution)
- [ ] Implement backtesting framework
- [ ] Add real-time data integration
- [ ] Multi-objective optimization

### Medium Priority

- [ ] Additional visualization options
- [ ] More portfolio constraints
- [ ] Performance optimization
- [ ] Additional risk metrics

### Documentation

- [ ] Video tutorials
- [ ] More examples
- [ ] Translation to other languages
- [ ] Jupyter notebooks

## Bug Reports

Please include:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error messages/stack traces

## Feature Requests

Please include:
- Clear description of feature
- Use case/motivation
- Example of how it would work
- Any relevant references

## Questions?

- Open an issue with the "question" label
- Check existing issues and documentation first

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project

Thank you for contributing! 🙏
