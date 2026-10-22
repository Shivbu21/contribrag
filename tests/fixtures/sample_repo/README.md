# SampleLib

A tiny sample library used as the ContribRAG test fixture. It does basic
arithmetic and token-based authentication for demo applications.

## Setup

Install the project in editable mode and run the test suite:

```bash
pip install -e .
python -m pytest tests/
```

## Usage

Import `add` and `divide` from `src/calc.py` for arithmetic, and
`hash_password` / `verify_token` from `src/auth.py` for authentication.
