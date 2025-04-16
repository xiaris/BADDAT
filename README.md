# ✨ BADDAT

Baseline-Aware Dependence fitting for DAmping Timescales (**BADDAT**) is a lightweight toolkit for forward modeling and statistical inference of AGN variability timescales and their dependence on physical parameters. It is particularly useful for addressing biases in DRW timescale measurements and exploring correlations with properties such as black hole mass, luminosity, and wavelength.

```python
import BADDAT
# Note that the input parameters should be numpy array
flat_samples = BADDAT.DependenceFitter1(tau, baseline, n_cadence, log_M_BH).fit()
...
```

![Demo](demo.png)

## 🚀 Getting Started

See `demo.ipynb` for a toy example demonstrating basic usage.


## 🔧 References

This project makes use of and adapts portions of code from open-source libraries [`taufit`](https://github.com/burke86/taufit) by Colin J. Burke –  parts of the functions for DRW likelihood evaluation and modeling were referenced and customized for this toolkit.

We thank the authors of these libraries for making their work openly available.




