# Microglia Morphology Analysis

This repository hosts the `atv_trem2_morpho` python package which reproduces the
single cell microglia morphology analysis in Extended Data Figure 2.

## Installing

Install the required versions of packages into a fresh python 3.10 virtual environment:

```{bash}
python3.10 -m venv path/to/env
source path/to/env/bin/activate
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt
```

This package has been tested on OS X 10.15 (Catalina), but may work on other
operating systems with minor modifications.

## Scripts

The `scripts/` directory defines the python scripts that reproduce the microglial
morphological analysis in Extended Figure 2 of the 2022 ATV:Trem2 paper:

* `plot_seg_stats.py`: Plot the results of a single cell morphology analysis

## Documentation

Package documentation is available in the `docs/` folder. Build and view the docs:

```{bash}
cd docs/
make html
open _build/html/index.html
```

## Tests

Package tests are available in the `tests/` folder. To run the tests in the virtual
environment:

```{bash}
python3 -m pytest tests/
```

Running the tests needs to be done from the base directory
