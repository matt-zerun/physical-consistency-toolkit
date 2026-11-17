# Release status: v0.2.0 (16 November 2026)

This release extends the previously validated v0.1.0 toolkit with independently
implemented EDA diagnostics inspired by Sections 2.3–2.4 of Energies 19(8),
1827. The PCI mathematical implementation and model estimation workflow are
retained, aside from the optional EDA report. No GitHub upload was performed by preparing the source archive.

**EDA does not reproduce empirical results**: the original laboratory data,
real-project input schemas and published numerical outputs are not embedded.
See `VALIDATION.md` and `docs/eda.md` for test evidence and scientific limits.

| Acceptance check | Status |
|---|---|
| Python syntax and package imports | Test evidence in VALIDATION.md |
| EDA + existing PCI regression tests | Test evidence in VALIDATION.md |
| Synthetically generated demonstration | Test evidence in VALIDATION.md |
| Data-independent public scheme | PASS by source design; original private schema excluded |
| No actual records or private model artifacts | Public archive content inspected; no such materials deliberately included |
| EDA non-destructive anomaly detection | Verified by integration tests |
| External-user CSV/XLSX, CLI and config | Validated locally; Windows GUI not separately verified |
| Historical table/figure reproduction | NOT VERIFIED (deliberately excluded) |
| Historical exact PCI values | NOT VERIFIED (unchanged prior release limitation) |
| Legal authority for code publication | Author declaration recorded; author should confirm final upload clearance |

Software copyright holder: Mateusz Żeruń. MIT software license. PIG-PIB appears
only as a scientific affiliation, not as an owner or endorser.

**GitHub packaging revision (2026-11-17):** Deposit-only metadata removed; computational code and tests unchanged.
