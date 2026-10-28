# Operational priority scoring

The operational layer converts row-level diagnosis predictions into grouped events and a transparent 0–100 priority score. It is deliberately separate from model training: severity, duration and equipment criticality are policy inputs, not learned model outputs.

```text
score = 100 × (0.45 × severity/5 + 0.30 × duration/5 + 0.25 × criticality/5)
```

The default policy is in `configs/operational_priority.yaml`. It maps recognised fault types to severity, maps labelled event duration to a 1–5 score, and assigns illustrative building criticality values. The criticality values must be reviewed against Smart Facilities SA's asset register before operational use.

Rows are grouped by building and AHU when adjacent timestamps are no more than the configured gap apart. Each event records its start and end timestamps, labelled duration, diagnosis confidence, the three component scores, total score, priority band and recommended action.

The current dataset labels complete fault sessions rather than the physical onset moment. Therefore, the duration is explicitly reported as **labelled event duration**. It must not be presented as proof that a physical fault began at the first timestamp or lasted until the last timestamp.

Run it with:

```bash
score_ahu_priorities \
  --predictions results/diagnosis/diagnosis_predictions.csv \
  --policy configs/operational_priority.yaml \
  --output-dir results/operational
```

The output includes CSV events and band counts, a JSON manifest, and a Markdown summary. The JSON manifest preserves the input, policy, design and duration limitation for auditability.
