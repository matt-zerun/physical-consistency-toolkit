# GitHub upload guide

1. Confirm the publication rights and read `NOTICE.md`, `THIRD_PARTY.md`, and `VALIDATION.md`. The author and copyright holder is Mateusz Żeruń; the included software license is MIT.
2. Create an empty GitHub repository, for example `physical-consistency-toolkit`. Do not initialize it with an additional README or license.
3. Extract `PCI_GITHUB_RELEASE_v0.2.0_CLEAN_20261117.zip`. Upload the **contents of the `physical-consistency-toolkit/` directory** as the repository root. Keep `.github/`, `.gitignore`, `CITATION.cff`, and the tests. Do not upload the ZIP as one file in place of a source tree.
4. From the repository root, install and test the program:

   ```bash
   python -m pip install ".[xlsx]"
   python -m unittest discover -s tests -v
   python -m physical_consistency eda --config examples/eda_demo.toml
   python -m physical_consistency demo --output my_synthetic_demo
   ```

5. Review `docs/eda.md`, `docs/mathematics.md` and `VALIDATION.md` for limitations. The included examples are synthetic and do not reproduce empirical tables or historical model scores.
6. Inspect the uploaded tree for unintended private content, local result directories, generated models or historical Git metadata. Start with a fresh repository history. Once confirmed, create a `v0.2.0` release tag.
7. When a real repository URL is available, update `README.md` and `CITATION.cff` with that URL. Do not use the methodological article DOI as a software DOI.

This repository was prepared for upload but was not uploaded automatically. The v0.2.0 source-code and tests are unchanged from the validated 2026-11-16 release; this 2026-11-17 packaging revision only removes deposit-specific metadata and updates GitHub documentation/checksums.
