# CrossCity-TrafficFM

Repository: https://github.com/ismailzrigui/crosscity-trafficfm

This public repository is kept as a code and governance package only.

## Public Scope

- Reproducible pipeline code under `scripts/`.
- Environment files: `requirements.txt` and `environment.yml`.
- Source-governance and reproduction notes in Markdown.
- Data-source metadata files such as `source_registry.md` and `data_manifest.md`.

## Local-Only Scope

The manuscript, PDF, DOCX, generated figures, generated tables, rendered pages, extracted text, and result reports are intentionally local only. They are ignored by Git through:

- `paper/`
- `results/`
- `data/raw/`
- `data/interim/`
- `data/processed/`

Do not publish article drafts or generated results to GitHub unless the author explicitly requests that in a later release step.

## Reproduction

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the local pipeline and tests:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
.\.venv\Scripts\python.exe scripts\run_tests.py
```

Manuscript builds and generated results remain on the local machine.
