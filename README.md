# AK-based-sampling

Python package for active learning-based reliability analysis and rare-event sampling methods, including AK-MCS and related surrogate-assisted approaches.

This repository provides organized, reusable, and extensible implementations for research and experimentation in uncertainty quantification and structural reliability analysis.

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/).

Create or update the local environment from the lockfile:

```powershell
uv sync
```

To run the example notebook with plotting dependencies:

```powershell
uv sync --extra examples
```

Run Python commands inside the uv-managed environment:

```powershell
uv run python -c "from ak_based_sampler import AK_MCS; print(AK_MCS)"
```

Launch Jupyter for the example notebooks after installing the example dependencies:

```powershell
uv run jupyter lab
```

## Usage

```python
import numpy as np
from ak_based_sampler import AK_MCS


def limit_state_function(u):
    return np.minimum.reduce([
        3 + 0.1 * (u[..., 0] - u[..., 1])**2 - (u[..., 0] + u[..., 1]) / np.sqrt(2),
        3 + 0.1 * (u[..., 0] - u[..., 1])**2 + (u[..., 0] + u[..., 1]) / np.sqrt(2),
        (u[..., 0] - u[..., 1]) + 6 / np.sqrt(2),
        (u[..., 1] - u[..., 0]) + 6 / np.sqrt(2),
    ])


sampler = AK_MCS(dim=2, LSF=limit_state_function)
pf = sampler(seed=1, verbose=True)

print(pf)
print(sampler.result)
```

## Examples

Example notebooks are available in the `examples/` directory.

- `examples/ak_mcs_basic.ipynb`: basic AK-MCS run with a sampling-result plot

If a notebook raises `ModuleNotFoundError: No module named 'ak_based_sampler'`, run `uv sync --extra examples` from this repository root, or restart the notebook kernel and run the first cell again.

## Implemented methods

This package currently implements the following AK-based reliability methods.
When you add a new method, add it to this table with the paper that the implementation follows.

| Class | Method | Reference |
| --- | --- | --- |
| `AK_MCS` | Active learning reliability method combining Kriging and Monte Carlo simulation | Echard et al. (2011) |
| `AK_IS` | Active learning reliability method combining Kriging and importance sampling | Echard et al. (2013) |

## Citation

If this package helps your work, please cite both the software and the papers that the implemented methods are based on.

Method reference template:

```bibtex
@article{echard2011ak,
  title={AK-MCS: an active learning reliability method combining Kriging and Monte Carlo simulation},
  author={Echard, Benjamin and Gayton, Nicolas and Lemaire, Maurice},
  journal={Structural safety},
  volume={33},
  number={2},
  pages={145--154},
  year={2011},
  publisher={Elsevier}
}
@article{echard2013combined,
  title={A combined importance sampling and kriging reliability method for small failure probabilities with time-demanding numerical models},
  author={Echard, Benjamin and Gayton, Nicolas and Lemaire, Maurice and Relun, Nicolas},
  journal={Reliability Engineering \& System Safety},
  volume={111},
  pages={232--240},
  year={2013},
  publisher={Elsevier}
}
```
