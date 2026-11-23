# Evaluation results

Generated artifacts from `python -m app.eval.run` land here.

Each run writes two files with a shared timestamp:

- `eval_<timestamp>.json` — the full report: dataset provenance, aggregation
  version, per-case predictions, metrics and the confusion matrix.
- `eval_<timestamp>.csv` — one row per case, for spreadsheet inspection.

**These files are generated, not source.** They are git-ignored, so running the
harness never dirties the working tree. Only this README and `.gitkeep` are
tracked.

To reproduce a run:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.eval.run
```

No API keys are required and no network call is made.
