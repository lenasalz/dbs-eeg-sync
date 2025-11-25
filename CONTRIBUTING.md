# Contributing to EEG-DBS Synchronization Toolbox

Thank you for your interest in contributing to the EEG-DBS Synchronization Toolbox! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

- **Bug reports**: Open an issue describing the problem, including steps to reproduce
- **Feature requests**: Suggest new features or enhancements via issues
- **Code contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve README, docstrings, or add examples
- **Testing**: Add or improve unit tests

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/dbs-eeg-sync.git
cd dbs-eeg-sync
```

### 2. Create Development Environment

We recommend using `uv` for fast, reproducible environments:

```bash
# Install dependencies
uv sync

# Install development dependencies
uv pip install -e ".[dev]"

# Optional: Install GUI dependencies
uv pip install -e ".[gui]"

# Optional: Install notebook dependencies
uv pip install -e ".[notebooks]"
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Code Style Guidelines

### General Principles

- **PEP 8 compliance**: Follow Python's PEP 8 style guide
- **Modern Python**: Use Python 3.12+ features (type hints with `|`, f-strings, etc.)
- **Type hints**: Add type annotations to all function signatures
- **Docstrings**: Use NumPy-style docstrings for all public functions/classes

### Type Hints

Use modern type hint syntax:

```python
# ✅ Good - modern style
def process_data(signal: np.ndarray, fs: float | None = None) -> tuple[np.ndarray, float]:
    ...

# ❌ Avoid - old style
from typing import Optional, Tuple
def process_data(signal: np.ndarray, fs: Optional[float] = None) -> Tuple[np.ndarray, float]:
    ...
```

### String Formatting

Use f-strings consistently:

```python
# ✅ Good
logger.info(f"Loaded {n_channels} channels at {fs:.2f} Hz")

# ❌ Avoid
logger.info("Loaded %d channels at %.2f Hz", n_channels, fs)
logger.info("Loaded {} channels at {} Hz".format(n_channels, fs))
```

### Imports

```python
# Standard library
from __future__ import annotations
from pathlib import Path
import logging

# Third-party
import numpy as np
import mne

# Local
from dbs_eeg_sync.data_loader import load_eeg_data
```

### Logging

Use appropriate log levels:

```python
logger.debug("Detailed debugging information")
logger.info("General informational messages")
logger.warning("Warning messages for unexpected but handled situations")
logger.error("Error messages for serious problems")
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dbs_eeg_sync --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestCore::test_sync_run_smoke
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Include docstrings explaining what is being tested

```python
def test_load_eeg_with_valid_file():
    """Test that load_eeg_data successfully loads a valid .set file."""
    file_path = Path("data/eeg_example.set")
    raw, fs = load_eeg_data(file_path)
    
    assert raw is not None
    assert fs > 0
    assert raw.n_times > 0
```

## 📚 Documentation

### Docstrings

Use NumPy-style docstrings:

```python
def synchronize_signals(
    eeg_data: mne.io.Raw,
    dbs_data: pd.DataFrame,
    method: str = "artifact"
) -> tuple[mne.io.Raw, pd.DataFrame]:
    """
    Synchronize EEG and DBS signals using the specified method.

    Parameters
    ----------
    eeg_data : mne.io.Raw
        The EEG data to synchronize.
    dbs_data : pd.DataFrame
        The DBS data to synchronize.
    method : str, default="artifact"
        Synchronization method: "artifact" or "manual".

    Returns
    -------
    tuple[mne.io.Raw, pd.DataFrame]
        Synchronized EEG and DBS data.

    Raises
    ------
    ValueError
        If method is not recognized.

    Examples
    --------
    >>> eeg, dbs = synchronize_signals(eeg_raw, dbs_df)
    >>> print(f"Synced {eeg.n_times} samples")
    """
    ...
```

## 🔄 Pull Request Process

### Before Submitting

1. **Run tests**: Ensure all tests pass
2. **Check linting**: Code should be clean (consider using `ruff`)
3. **Update docs**: Add/update docstrings and README if needed
4. **Add tests**: Include tests for new features
5. **Update CHANGELOG**: Add entry describing your changes (if applicable)

### PR Guidelines

- **Clear title**: Use descriptive PR titles (e.g., "Add support for BDF file format")
- **Description**: Explain what changed and why
- **Link issues**: Reference related issues (e.g., "Fixes #42")
- **Small PRs**: Keep changes focused and manageable
- **Commits**: Use clear, descriptive commit messages

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Tests pass locally
- [ ] Added/updated tests
- [ ] Updated documentation
- [ ] Code follows style guidelines
```

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Clear title**: Descriptive summary of the issue
- **Steps to reproduce**: Minimal example to reproduce the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, package versions
- **Error messages**: Full traceback if applicable

### Feature Requests

Include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches you've considered

## 🤝 Code of Conduct

### Our Standards

- **Be respectful**: Treat everyone with respect and kindness
- **Be constructive**: Provide helpful feedback
- **Be collaborative**: Work together towards common goals
- **Be inclusive**: Welcome contributors of all backgrounds and skill levels

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling, insulting, or derogatory remarks
- Publishing others' private information

## 📬 Questions?

- **GitHub Issues**: For bugs and feature requests
- **Email**: [lena.salzmann@hest.ethz.ch](mailto:lena.salzmann@hest.ethz.ch) for general questions

## 📄 License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.

---

Thank you for contributing to the EEG-DBS Synchronization Toolbox! 🎉

