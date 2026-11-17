# Scientific and technical validation

**Release:** Physical Consistency Toolkit **v0.2.0** (prepared 2026-11-16).

## Executed validation for this release

- **59/59 Python `unittest` tests PASS** on Linux, Python 3.13.5, no skipped tests,
  using `PYTHONPATH=src python -m unittest discover -s tests -v`.
- Built a pure-Python **v0.2.0 wheel** using `pip wheel --no-build-isolation --no-deps`.
  Installed the wheel in a separate target directory. Standalone `python -m
  physical_consistency` commands executed outside the source tree using the
  host runtime's already installed dependencies.
- Standalone EDA TOML run completed on **225 newly generated synthetic records**:
  45 below an illustrative 1% threshold and 180 at-or-above; generated CSV
  diagnostics and PNG figures. Outlier labels were counted from flags, not
  total tested rows. No records were removed from the source.
- Full synthetic offline demo completed on **240 independently generated
  records**, evaluating SVR, RF, Decision Tree and KNN, exporting metrics,
  response surfaces, EDA tables and plots. All results are intentionally
  illustrative, not article reproductions.
- Analytic EDA tests cover sample SD, IQR, CV, skewness, excess kurtosis,
  Pearson correlation, intercept-corrected VIF including exact collinearity,
  standardized PCA, optional threshold segmentation, non-destructive outlier
  flags, public column mapping, validation policy and CLI paths.
- The **pre-existing PCI analytical tests** continue to pass unchanged,
  including monotonicity, roughness, boundaries, grid masks, synthetic model
  fitting, interactive output and generated CSV formats.
- Source ZIP audit/manifest verification is reported in the accompanying
  separately in the release QA note and external archive SHA-256 checksums;
  repository file hashes are listed in `MANIFEST_SHA256.txt`.

## Scientific limitations

The paper by Żeruń, Jagoda and Majer (Energies 19(8), 1827, DOI
10.3390/en19081827) provides the methodological basis for the EDA analytical
categories, PCI equations and model evaluation scope. This release implements
those **types of analyses** with independently written code.

**NOT VERIFIED:** exact reconstruction of the original manuscript's empirical
Table 1, Figures 1–3, the original raw-data selection or historic numerical
PCI model scores. The original measurements, private project schema, model
artifacts and internal system are not included. Synthetic tests cannot
establish historical equivalence.

PCA and k-means clusters are descriptive only; fitted PCA and cluster
assignments are never passed to ML. Isolation Forest provides anomaly flags,
not automated exclusions. CV is sample SD divided by absolute sample mean.
VIF is calculated using an intercept; the method and numerical results may
not coincide with uncentred calculations found in historical scripts.

The published manuscript does not uniquely define every PCI finite-difference
boundary stencil. This independent implementation retains the disclosed
numerical conventions of v0.1.0, as explained in docs/mathematics.md.
Physical-consistency rankings remain relative to the model cohort and domain.

## Environment and remaining checks

- **Tested:** Linux, Python 3.13.5, current installed package environment.
- **Target, not re-tested for v0.2.0:** Windows/Spyder and Python 3.11.5.
  Prior v0.1.0 tests on Python 3.11 and 3.12 are historical evidence, not a
  substitute for executing the new EDA version there. GitHub Actions is
  configured for Linux/Windows Python 3.11/3.12 after publishing the repository.
- No real user data and no private database structures were used for v0.2.0
  release tests. No GitHub repository was created.
- Third-party dependencies are not vendored; MIT covers toolkit source only.
  Read NOTICE.md / THIRD_PARTY.md before publication.

**GitHub packaging revision (2026-11-17):** Removes deposit-specific metadata and corrects GitHub documentation. The computational source files and test suite are byte-identical to the tested v0.2.0 archive; this packaging revision has no new scientific validation claims.
