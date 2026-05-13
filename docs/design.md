# sedinfer Design Notes

## Package Architecture

`sedinfer` is split around stable interfaces rather than physical model details:

- `data.py` contains observed-data containers.
- `priors.py` and `parameters.py` define scalar priors and the ordered parameter vector contract.
- `filters.py` contains a thin `FilterSet` wrapper for backend filter objects.
- `backends/` contains forward-model implementations behind a common interface.
- `likelihood.py` evaluates backend-agnostic photometric likelihoods.
- `transforms/` contains physical or catalog-specific parameter transforms, such as SFH utilities.
- `runners.py` provides light glue to inference tools.

The likelihood and parameter-space layers should stay small and deterministic. Backend-specific physics belongs in backends or transforms.

## Backend Contract

Every backend exposes:

```python
predict_photometry(params, filters) -> ModelPhotometry
```

`ModelPhotometry.flux` is a one-dimensional linear flux vector. `ModelPhotometry.band_names` names every element so the likelihood can align model and observed bands without assuming positional order.

Backends must also declare:

```python
mass_normalization: MassNormalization
```

Backends may use any internal physics library, but they should raise ordinary Python numerical exceptions for invalid model states and return finite model fluxes for valid states.

## Mass Normalization

Mass scaling is explicit. A backend must choose one of:

- `MassNormalization.PER_SOLAR_MASS`: model fluxes are per solar mass formed. The likelihood requires a `log10_mass` parameter and multiplies model flux by `10**log10_mass`.
- `MassNormalization.ABSOLUTE`: model fluxes are already absolute. The likelihood never applies `log10_mass`, even if that parameter exists.

The likelihood never infers this behavior from parameter names, backend class names, or flux magnitudes.

## Physical Transforms

Physical transforms live under `sedinfer/transforms/`. For example, Pop-COSMOS continuity-SFH utilities convert catalog theta rows into FSPS-ready tabular SFHs.

Transforms should be pure functions where possible. They should not own backend instances, caches, multiprocessing pools, or global state.

## Backend-Agnostic Likelihood

`likelihood.py` must remain backend-agnostic because it defines the statistical contract:

- evaluate `log_prior + Gaussian log likelihood`,
- apply masks consistently to fluxes and uncertainties,
- align photometry by band name,
- apply only declared mass normalization,
- return `-inf` for controlled backend numerical failures,
- raise clear errors for API/configuration mismatches.

Backend-specific parameter aliases such as `z`, `zred`, or `redshift` must be handled by transforms or backends, not by the likelihood.
