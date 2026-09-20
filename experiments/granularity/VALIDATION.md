# Validation and interpretation audit

The complete fixed grid ran successfully: five designs, three calibration sizes,
30 seeds, seven methods, and two reporting precisions. The evaluation population
contains all 1,000 score types. Retained predictions permit reevaluation without
refitting. Configuration and source/result hashes identify the run.

An independent reviewer checked the mathematical identities and allocation rules
before the full run. Additional independent checks on 100 random populations used
exact integration over decision breakpoints. Post-run review checked every summary
mean and paired interval against the evaluation archive. Two prose corrections
were incorporated: constant-risk maps cannot lose risk information, and the
miscalibration component equals twice the integrated regret gap.

Local validation passed:

- Full repository suite: 1,709 passed, four existing R-reference skips.
- Final experiment suite after adding five focused checks: 19 passed.
- Ruff checks and formatting, pydoclint on the package, and pyright.
- Standard `make granularity` and `make granularity-check` targets.
- Notebook execution with 23 embedded SVG figures and no error outputs.
- LaTeX compilation without unresolved citations or overfull boxes.
- Visual inspection of all 22 original PDF pages, including every table block.
  The bibliography revision produces 24 pages; changed prose and reference pages
  were rendered and inspected again.
- Full-grid decomposition, rounding/coarsening, and regret-order inequalities.
- Fit-time source hashes and exact reproduction of the saved seed-zero pilot.

Intervals describe Monte Carlo uncertainty in expected performance across
calibration samples. They do not describe the predictive distribution of the
performance of one fitted calibrator. No simultaneous curve dominance is claimed.
The structural diagnostics and oracle information loss cannot by themselves show
that a user can recover the retained information.

There were no changes to the prespecified population, sample sizes, methods,
seeds, cost grid, or estimands after results were inspected. Presentation changes
removed a mostly empty page, moved overlapping legends, and fixed the capacity
axis for the constant-risk design so numerical noise is not magnified. Figures
retain all methods and designs; rounding sensitivity is reported separately.

The bibliography revision adds nine cited sources in `references.bib`, identifies
grouping loss and the value-of-information argument as established foundations,
and limits the contribution to the reproducible decision comparison. The strict
LaTeX/BibTeX build and all 19 experiment tests passed again after this revision.

## Score-replacement theory revision

The revised note includes four propositions with proofs, a claim-to-precedent
comparison, and an explicit methods-note judgment. The original simulation grid
and numerical estimands remain unchanged. The existing results were already
known when this theory and positioning review began.

An independent reviewer checked the threshold conditions, finite-population and
general-distribution claims, allocation equivalence, and known-inverse argument.
Three clarifications were incorporated: probability-valued reports, the exact
costly-error event under a strict threshold rule, and the support of the threshold
weighting distribution. The reviewer did not independently audit source metadata;
source versions and titles were checked against their primary records separately.

Validation after this revision:

- 23 experiment tests passed, including four new boundary and decoding checks.
- Repository Ruff lint and formatting checks passed (115 files).
- `make granularity` rebuilt figures, summaries, and the executed notebook.
- The final strict LaTeX/BibTeX build passed with no unresolved citations or
  overfull boxes. All 27 pages were rendered and visually inspected.
- The literature table separates established results, specializations, and the
  unresolved finite-sample recoverability question. No priority claim is made.

## Paper-first economic rewrite

The main argument now occupies six pages: the decision problem, estimation
commitments, information and implementation losses, selected existing evidence,
and how to judge a procedure. Proofs, the contribution audit, complete design,
all original exhibits, and tables are retained in appendices. The final PDF has
32 pages including references. Every page was rendered and visually inspected;
a short overflow page in the main evidence section was removed during layout QA.

The added PAV citation supports empirical optimality for regular binary proper
scoring rules. The manuscript explicitly limits that statement to constrained
fitted values and separates it from out-of-sample decision performance. Numerical
claims continue to use generated macros and existing exhibits. No simulation
result, estimator, package API, or separate software manuscript changed in this
revision. `PACKAGE_PROPOSAL.md` records proposed operations and requires author
review before package implementation.

The full `make granularity` target passed, including notebook execution and strict
LaTeX/BibTeX compilation. All 23 focused tests and repository Ruff lint/format
checks passed. No full repository test rerun was needed for this prose/layout-only
revision. The previous full-suite result above remains historical.
