# Choosing calibration for decisions

The research note starts from costs and capacity constraints, explains what
calibration estimates, and evaluates the fitted map, reported information, and
action rule together. The main text uses selected existing exhibits; proofs,
claim-to-literature assessment, and exhaustive results are in appendices.
`theory.tex` contains the proofs and contribution assessment.

[PACKAGE_PROPOSAL.md](PACKAGE_PROPOSAL.md) maps the argument to existing package
capabilities and records the additive implementation. The public decision API
now evaluates costs and capacity, selects on validation predictions, and evaluates
frozen policies on disjoint test cases. See the
[worked example](../../docs/examples/decisions.rst).

The existing simulation measures useful distinctions lost through calibration, then
separates that loss from error in reported probability levels. Its metrics use
known population risks; they are not public estimators from predictions and labels.

From the repository root:

```bash
uv sync --locked --all-groups
make granularity-run
make granularity
make granularity-check
```

`granularity-run` fits all 450 calibration samples (seven methods each) and saves
predictions and evaluations. `granularity` generates figures, CSV summaries, an
executed notebook, and `experiments/granularity/build/note.pdf`. It requires
`latexmk` and a LaTeX installation with BibTeX. References are maintained in
`references.bib`; `latexmk` resolves citations automatically. Open `granularity.ipynb` to inspect or rerun the
analysis interactively. Builds do not refit methods.

For a numerical evaluation change with the same fitted populations and methods:

```bash
uv run python -m experiments.granularity.study --evaluate-only
```

The saved configuration must match. For changes to methods or calibration
sampling, refit instead. Do not overwrite fit-time provenance to make old results
look current.

## Outputs

- `results/predictions.npz`: design × size × seed × method × population type.
- `results/evaluation.npz`: metrics, threshold curves, and capacity curves.
- `results/config.json`, `environment.json`, `manifest.json`: design, source hashes,
  versions, and result checksums. The manifest names array axes and measurements.
- `build/metrics.csv`: every replicate and precision, including ordering diagnostics.
- `build/summary.csv`: means and paired differences with bootstrap intervals.
- `build/granularity.ipynb`: executed notebook; source notebook stays output-free.
- `build/note.pdf`: standalone explanation, all design figures and complete tables.

The population is exactly the finite set of 1,000 types; conditional risk is
computed over that full set, not estimated from isolated sample predictions.
Spearman is unavailable for constant forecasts. Bootstrap intervals describe
Monte Carlo uncertainty in mean performance across calibration samples, not the
range of outcomes for an individual fitted calibrator. They are pointwise and
unadjusted for multiplicity.

`DESIGN.md` records the pre-run choices and prior exposure to other benchmarks.
The existing manuscript and benchmark results are not changed by these targets.
