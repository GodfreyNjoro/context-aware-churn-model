# Contributing to Context-Aware Churn Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions.

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch** from `develop`
3. **Make your changes**
4. **Write or update tests**
5. **Ensure all tests pass**
6. **Submit a pull request**

## Development Setup

```bash
git clone https://github.com/GodfreyNjoro/context-aware-churn-model.git
cd context-aware-churn-model
./setup.sh
```

## Coding Standards

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all public functions
- Maintain test coverage above 80%
- Run linters before committing:
  ```bash
  black src/
  isort src/
  flake8 src/
  ```

## Testing

```bash
pytest tests/ --cov=src
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure CI passes
4. Request review from maintainers

## Questions?

Open an issue or contact the maintainers.
