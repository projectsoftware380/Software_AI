# Software_AI

This repository contains the sources for the trading pipeline.  The Python
package uses a `src` layout so all modules live under the `src/` directory.

## Installation

1.  Clone the repository and change into its directory.
2.  Install the package using `pip`.  All runtime dependencies from
   `requirements.txt` are declared in `pyproject.toml`.

```bash
pip install .
```

This installs the `src` package along with all required third party
libraries.

## Running tests

After installing the package you can run the unit tests with `pytest`:

```bash
pytest
```


For details on the critical bug fixed in pipeline v5, see [docs/Pipeline_v5_error_fix.md](docs/Pipeline_v5_error_fix.md).
