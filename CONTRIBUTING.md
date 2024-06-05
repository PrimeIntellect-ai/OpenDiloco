# Development workflow

This is the develpment workflow of prime intellect to build upon hivemind

## Install dependencies

Install hivemind  

```bash
cd hivemind_source
pip install .
cp build/lib/hivemind/proto/* hivemind/proto/.
pip install -e ".[all]"```
```

## Pre-commit hook

Install the pre commit hook to keep black and isort updated on each commit:

```
pre-commit install
```

## Testing
To run the tests:

```
python -m pytest tests
```

Be sure to actually use python -m otherwise path won't be appended correctly

# Development flags
Add the `PRIME_INTELLECT_DEV` environment variable to your *.bashrc* or *.zshrc* so that development features are enabled.
e.g.
- torch compile error will crash the script instead of silently failing
