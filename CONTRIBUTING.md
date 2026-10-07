# Contributing

This repository is maintained as a research and portfolio implementation of ICU deterioration prediction using credentialed MIMIC-IV data.

## Development checks

Before opening a pull request:

```bash
python -m compileall -q app.py run_pipeline.py src
ruff check app.py run_pipeline.py src --select E9,F63,F7,F82
```

Do **not** commit MIMIC-IV raw data, derived patient-level datasets, credentials, service-account keys, or generated model artifacts containing patient-level information. The `.gitignore` is configured to exclude the local data directories and common credential files.

Changes that affect cohort construction, observation windows, outcome definitions, imputation, train/test splitting, or calibration should be documented because they can materially change reported performance.
