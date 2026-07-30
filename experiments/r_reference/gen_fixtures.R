#!/usr/bin/env Rscript
# Generates cross-language reference fixtures for calibre's core estimators.
#
# Fixtures are committed to tests/fixtures/r/ so CI needs no R installation.
# Regenerate deliberately:  Rscript experiments/r_reference/gen_fixtures.R
#
# Required packages: isotone, Iso, cir, neariso, scam
#
# neariso (CRAN-archived, 2011) does not compile against modern libc++ because
# R.h's `length` macro expands inside libc++ <locale>. To build it:
#   curl -O https://cran.r-project.org/src/contrib/Archive/neariso/neariso_1.0.tar.gz
#   tar xzf neariso_1.0.tar.gz
#   # in neariso/src/NIR.cc, move <vector>/<map>/<math.h> above <R.h>
#   # and add `#undef length` after the R includes
#   R CMD INSTALL neariso

suppressPackageStartupMessages({
  library(scam)
  library(isotone)
  library(Iso)
  library(cir)
  library(neariso)
  library(jsonlite)
})

out_dir <- file.path("tests", "fixtures", "r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------- test cases --
# Deliberately includes the degenerate shapes where calibre currently breaks:
# heavy ties in x, all-zero and all-one labels, n=1, n=2, constant x.
cases <- list(
  simple        = list(x = 1:10,
                       y = c(0.10,0.40,0.20,0.35,0.30,0.65,0.55,0.90,0.75,1.00)),
  binary        = list(x = 1:12,
                       y = c(0,0,1,0,0,1,1,0,1,1,1,1)),
  ties_in_x     = list(x = c(1,1,1,2,2,3,3,3,3,4),
                       y = c(0,1,0,1,1,0,1,1,1,1)),
  all_zero      = list(x = 1:6,  y = rep(0, 6)),
  all_one       = list(x = 1:6,  y = rep(1, 6)),
  decreasing    = list(x = 1:8,  y = c(1,1,1,0.8,0.5,0.2,0,0)),
  n1            = list(x = 1,    y = 1),
  n2            = list(x = c(1,2), y = c(1,0)),
  constant_x    = list(x = rep(2, 5), y = c(0,1,1,0,1)),
  weighted      = list(x = 1:8,
                       y = c(0.2,0.9,0.1,0.5,0.4,0.8,0.6,1.0),
                       w = c(1,5,1,3,1,1,7,2))
)

fixtures <- list()

for (nm in names(cases)) {
  cs <- cases[[nm]]
  x <- as.numeric(cs$x); y <- as.numeric(cs$y)
  w <- if (is.null(cs$w)) rep(1, length(y)) else as.numeric(cs$w)
  entry <- list(x = x, y = y, w = w)

  # --- unweighted PAVA (two independent references) ----------------------
  entry$isoreg <- as.numeric(stats::isoreg(x, y)$yf)
  entry$iso_pava <- as.numeric(Iso::pava(y, w = rep(1, length(y))))

  # --- weighted PAVA -----------------------------------------------------
  # gpava with weighted.mean is the generalized PAVA of de Leeuw et al.
  # (solver is base R's stats::weighted.mean, gpava's default)
  gp <- isotone::gpava(z = seq_along(y), y = y, weights = w,
                       solver = weighted.mean, ties = "primary")
  entry$gpava_weighted <- as.numeric(gp$x)
  entry$iso_pava_weighted <- as.numeric(Iso::pava(y, w = w))

  # --- centered isotonic regression (CIR) --------------------------------
  # cirPAVA collapses PAVA's flat blocks to single points and interpolates.
  # It returns one fitted value per *unique* x, so record that grid alongside.
  entry$cir_x <- sort(unique(x))
  entry$cir <- tryCatch({
    as.numeric(cir::cirPAVA(y = y, x = x, wt = w, full = FALSE))
  }, error = function(e) NULL)

  # Aggregated (unique-x) inputs, which is what any tie-aware fit must use.
  agg_w <- as.numeric(tapply(w, x, sum))
  entry$agg_x <- as.numeric(sort(unique(x)))
  entry$agg_w <- agg_w
  entry$agg_y <- as.numeric(tapply(y * w, x, sum)) / agg_w

  fixtures[[nm]] <- entry
}

# ------------------------------------------------- nearly-isotonic solutions --
# Paper objective (Tibshirani, Hoefling & Tibshirani 2011, Technometrics 53(1)):
#     (1/2) * sum (y_i - b_i)^2  +  lambda * sum max(0, b_i - b_{i+1})
# Recorded under `lambda_paper`. A package using an un-halved SSE term needs
# lambda = 2 * lambda_paper to reach the same solution.
#
# LIMITATION: neariso() aborts with "'breaks' are not unique" whenever two
# merges fall on the same lambda -- which is the norm for binary labels and for
# monotone-decreasing input. So the reference set here is continuous data in
# general position only. On those cases neariso pins calibre's CVXPY solver
# exactly (~1e-16); the validated CVXPY solver is then what we use as the
# reference for binary labels, where neariso cannot run.
#
# Targets are kept inside [0,1] on purpose. calibre applies a non-optional
# np.clip(., 0, 1) to its output, so on data straying outside the unit interval
# it necessarily disagrees with neariso (which has no such clip) even when its
# solver is exactly right -- confirmed empirically: the only mismatches on
# out-of-range data were at the clipped endpoints. Mixing the two effects into
# one fixture would make these tests untrustworthy. The clip itself is pinned
# separately by test_clipping_is_opt_out_able.
set.seed(11)
# Affine rescale into [0.02, 0.98] rather than clipping: clamping creates tied
# values, and tied values make two merges land on the same lambda, which is the
# exact condition under which neariso aborts.
squash <- function(v) round(0.02 + 0.96 * (v - min(v)) / (max(v) - min(v)), 6)
ni_cases <- list(
  simple = c(0.10,0.40,0.20,0.35,0.30,0.65,0.55,0.90,0.75,1.00),
  wiggly = c(0.05,0.31,0.22,0.28,0.47,0.41,0.52,0.71,0.63,0.88,0.79,0.97),
  noisy  = squash(sort(runif(25)) + rnorm(25, 0, 0.13)),
  steep  = squash(plogis(seq(-4, 4, length.out = 20)) + rnorm(20, 0, 0.07))
)

# An explicitly out-of-range sequence, recorded so the clipping behaviour can be
# tested against an unclipped reference rather than inferred.
ni_unclipped <- round(plogis(seq(-4, 4, length.out = 16)) * 1.2 - 0.1, 6)

ni <- list()
for (nm in names(ni_cases)) {
  y <- as.numeric(ni_cases[[nm]])
  path <- tryCatch(neariso::neariso(y, maxBreaks = 500), error = function(e) NULL)
  if (is.null(path)) {
    cat("skipping neariso case", nm, "(simultaneous merges)\n")
    next
  }

  # The path object already carries exact solutions at every breakpoint, so take
  # those directly rather than going through nearisoGetSolution -- its internal
  # cut() fails with "'breaks' are not unique" whenever two merges occur at the
  # same lambda, which happens routinely on binary and monotone-decreasing data.
  sols <- list()
  for (j in seq_along(path$lambda)) {
    lam <- path$lambda[j]
    sols[[sprintf("bp%02d", j)]] <- list(
      lambda_paper = as.numeric(lam),
      beta = as.numeric(path$beta[, j]),
      df = as.numeric(path$df[j])
    )
  }

  # Additionally probe a few off-breakpoint lambdas where the solver permits it.
  for (lam in c(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)) {
    s <- tryCatch(neariso::nearisoGetSolution(path, lambda = lam),
                  error = function(e) NULL)
    if (!is.null(s)) {
      sols[[paste0("lam", format(lam, scientific = FALSE))]] <- list(
        lambda_paper = lam,
        beta = as.numeric(s$beta),
        df = as.numeric(s$df)
      )
    }
  }

  ni[[nm]] <- list(
    y = y,
    breakpoints = as.numeric(path$lambda),
    df_path = as.numeric(path$df),
    solutions = sols
  )
}

ni_oor <- list()
{
  y <- as.numeric(ni_unclipped)
  path <- tryCatch(neariso::neariso(y, maxBreaks = 500), error = function(e) NULL)
  if (!is.null(path)) {
    sols <- list()
    for (lam in c(0.05, 0.1, 0.25)) {
      s <- tryCatch(neariso::nearisoGetSolution(path, lambda = lam),
                    error = function(e) NULL)
      if (!is.null(s)) {
        sols[[paste0("lam", format(lam, scientific = FALSE))]] <- list(
          lambda_paper = lam, beta = as.numeric(s$beta))
      }
    }
    ni_oor <- list(y = y, solutions = sols)
  }
}

# --------------------------------------------------- monotone P-spline (scam) --
# scam's "mpi" basis is a shape-constrained P-spline (Pya & Wood 2015): a B-spline
# basis with a discrete coefficient penalty, reparameterised so the coefficients
# form an increasing sequence. That is the same construction calibre uses, so it is
# the right external reference for the monotone spline calibrator.
#
# Exact agreement is NOT expected and must not be asserted: scam selects its
# smoothing parameter by GCV while calibre cross-validates log-loss. What is
# comparable is the fitted curve (they agree to ~2e-3 in practice) and the
# in-sample log-loss.
set.seed(3)
scam_cases <- list()
for (nm in c("logistic", "concave", "steep")) {
  n <- 800
  x <- sort(runif(n))
  p <- switch(nm,
    logistic = plogis(3 * (x - 0.5)),
    concave  = sqrt(x),
    steep    = plogis(8 * (x - 0.5)))
  y <- rbinom(n, 1, p)
  fit <- tryCatch(
    scam(y ~ s(x, bs = "mpi", k = 10), family = binomial,
         data = data.frame(x = x, y = y)),
    error = function(e) NULL)
  if (is.null(fit)) {
    cat("skipping scam case", nm, "\n"); next
  }
  grid <- seq(min(x), max(x), length.out = 200)
  pred <- as.numeric(predict(fit, newdata = data.frame(x = grid), type = "response"))
  insample <- as.numeric(fitted(fit))
  scam_cases[[nm]] <- list(
    x = x, y = as.numeric(y), grid = grid, fitted_grid = pred,
    logloss = -mean(y * log(pmax(insample, 1e-12)) +
                    (1 - y) * log(pmax(1 - insample, 1e-12)))
  )
}

meta <- list(
  generated_by = "experiments/r_reference/gen_fixtures.R",
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  packages = sapply(c("isotone", "Iso", "cir", "neariso", "scam"),
                    function(p) as.character(utils::packageVersion(p)))
)

write_json(list(meta = meta, cases = fixtures),
           file.path(out_dir, "pava_reference.json"),
           digits = 17, auto_unbox = FALSE, pretty = TRUE, null = "null")
write_json(list(meta = meta, cases = ni, out_of_range = ni_oor),
           file.path(out_dir, "neariso_reference.json"),
           digits = 17, auto_unbox = FALSE, pretty = TRUE, null = "null")
write_json(list(meta = meta, cases = scam_cases),
           file.path(out_dir, "scam_reference.json"),
           digits = 17, auto_unbox = FALSE, pretty = TRUE, null = "null")

cat("wrote", file.path(out_dir, "pava_reference.json"), "\n")
cat("wrote", file.path(out_dir, "neariso_reference.json"), "\n")
cat("wrote", file.path(out_dir, "scam_reference.json"), "\n")
