# Operational priority scoring

The operational layer consumes row-level **binary detector predictions**, keeps only rows where the detector reports a fault, groups adjacent alerts into events, and calculates a transparent 0–100 priority score. It is deliberately separate from model training: severity, duration and equipment criticality are policy inputs, not learned model outputs.

```text
score = 100 × (0.45 × severity/5 + 0.30 × duration/5 + 0.25 × criticality/5)
```

The default policy is in `configs/operational_priority.yaml`. It maps recognised fault types to severity, maps detected-alert duration to a 1–5 score, and assigns illustrative building criticality values. The criticality values must be reviewed against Smart Facilities SA's asset register before operational use.

Diagnosis is optional. When a matching row-level diagnosis exists, its fault type and model score are attached; otherwise the event remains `Uncertain`. Matching uses `row_id` and, when available, the evaluation `design` and `fold`, so a diagnosis from another split cannot be attached accidentally. Rows are grouped by building, AHU and model/split context when adjacent timestamps are no more than the configured gap apart. Each event records its start and end timestamps, detected duration, detected row count, diagnosis model score, the three component scores, total score, priority band and recommended action.

The current dataset labels complete fault sessions rather than the physical onset moment. Therefore, the duration is explicitly reported as **detected-alert duration**. It describes the consecutive timestamps flagged by the detector; it must not be presented as proof that a physical fault began at the first timestamp or lasted until the last timestamp.

Run it with:

```bash
score_ahu_priorities \
  --predictions results/quality_controlled/exclude_ambiguous_keys/quality_detection_predictions.csv \
  --diagnosis-predictions results/diagnosis/diagnosis_predictions.csv \
  --model xgboost \
  --design chronological \
  --policy configs/operational_priority.yaml \
  --output-dir results/operational
```

The detector output is produced by `run_quality_detection` as `quality_detection_predictions.csv`; it contains one row per scored test observation, including both normal and fault predictions. The output includes CSV events and band counts, a JSON manifest, and a Markdown summary. The JSON manifest preserves the detector input, optional diagnosis input, policy, model, design and duration limitation for auditability. `y_true` is retained in the detector audit file but is not used to create or score operational events.
