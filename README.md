# Model Error Quantification for WNVI — The illusion of fit

Code accompanying

> Vincent C. Scholz, P.S. Koutsourelakis,
> **The illusion of fit: Spatially resolved assessment of constitutive model validity in elastography and physics-based inverse problems**,
> *Computer Methods in Applied Mechanics and Engineering*, Volume 461, Part B, 2026, 119208.
> https://doi.org/10.1016/j.cma.2026.119208

Inferring material properties from measured deformations usually assumes that the constitutive law is valid
everywhere, which yields plausible-looking parameter maps even when it is not (the "illusion of fit").
The method treats the stress field as an independent latent variable, so that mechanical equilibrium
(conservation law, small fixed precision) and the constitutive law (precision inferred under a
sparsity-promoting prior) enter as separate virtual observables. The inferred spatial map of constitutive
precisions shows where the assumed model is supported by the data and where it is not. Inference is
done with stochastic variational inference and is forward-model-free.

This repository reproduces the **synthetic harmonic brain-slice experiment at 35 dB noise** (linear elastic
domain with a localized anisotropic inclusion, 64x64 observation grid).

## Setup

Requires Python 3.11-3.13 and, practically, a CUDA GPU (the original runs used an H100).

```bash
poetry install
```

Versions are pinned to the environment used for the paper (torch 2.7.1, matplotlib 3.10.3, ...).
FEniCS is **not** needed: the forward-model solution (displacements and reference stresses) is provided in `data/`.
The reference stress is only used for error plots.

## Running the experiment

```bash
poetry run python run_inference.py configs/brain_35dB.py
```

A single config describes the whole experiment, which runs in two stages inside one process
(the posterior state and the noisy data are handed from stage 1 to stage 2 in memory):

| Stage | Iterations | Description |
|---|---|---|
| `stage1` | 1.75M | Trained from scratch; Exponential constitutive likelihood, one residual per element, `lmbda_c` frozen. |
| `stage2` | 1.75M | Starts from the stage-1 posterior (`lmbda_c` re-initialised); Gaussian constitutive likelihood with one precision per stress component and element, prior on `x` deactivated. |

The per-stage differences are listed in `stages` at the bottom of `configs/brain_35dB.py`.
Output (plots, logs, checkpoints every 50k iterations, a copy of the config) is written to
`results/<date>_#_<i>/<stage>/`.

Use `--seed N` to override the random seed (default `seed = 0` in the config; it fixes the noise realisation
and network initialisation). Seeded runs are not guaranteed to be bit-wise identical to the paper runs,
which were not seeded.

## Layout

- `run_inference.py` – driver: seeds, loops over the stages
- `stage_runner.py` – builds model, priors, likelihoods and posteriors and runs the optimisation of one stage
- `configs/` – experiment settings (geometry, priors, likelihoods, posteriors, optimiser, stages)
- `model/` – PDE, basis functions, distributions, ELBO / SVI
- `utils/` – initialisation, plotting, logging helpers
- `data/` – displacement data, reference stress, brain segmentation image

## Citation

```bibtex
@article{scholz2026illusion,
  title   = {The illusion of fit: Spatially resolved assessment of constitutive model validity in elastography and physics-based inverse problems},
  author  = {Scholz, Vincent C. and Koutsourelakis, P.S.},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {461},
  pages   = {119208},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.119208}
}
```

## License

MIT, see `LICENSE`.
