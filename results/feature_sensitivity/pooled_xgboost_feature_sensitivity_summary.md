# Pooled XGBoost feature sensitivity

The default BMS-signal model is the practical headline. Symptoms-only is a feature ablation, not a simple baseline.

| Variant | Design | Folds | Features | Mean F1 | Mean balanced accuracy | Mean FPR | Test rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default_bms_signals | building_held_out | 3 | 88 | 0.284 | 0.584 | 0.118 | 354,587 |
| symptom_only | building_held_out | 3 | 35 | 0.471 | 0.643 | 0.122 | 354,587 |
| default_bms_signals | chronological | 1 | 88 | 0.823 | 0.909 | 0.176 | 70,935 |
| symptom_only | chronological | 1 | 35 | 0.767 | 0.859 | 0.198 | 70,935 |
| default_bms_signals | nested_grouped | 5 | 88 | 0.956 | 0.965 | 0.006 | 354,587 |
| symptom_only | nested_grouped | 5 | 35 | 0.941 | 0.956 | 0.020 | 354,587 |
