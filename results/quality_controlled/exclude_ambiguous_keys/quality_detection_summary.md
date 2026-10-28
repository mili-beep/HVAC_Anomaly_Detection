# Quality-controlled pooled detection summary

All models use the declared physical-signal feature variant. Imputation and scaling are fitted within each training fold.

| Policy | Design | Model | Mean F1 | Mean balanced accuracy | Mean FPR | Mean coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| exclude_ambiguous_keys | ahu_held_out | xgboost | 0.971 | 0.972 | 0.002 | 1.000 |
| exclude_ambiguous_keys | ahu_held_out | normal_range_rule | 0.539 | 0.650 | 0.271 | 1.000 |
| exclude_ambiguous_keys | ahu_held_out | isolation_forest | 0.106 | 0.525 | 0.006 | 1.000 |
| exclude_ambiguous_keys | ahu_held_out | lstm_autoencoder | 0.024 | 0.502 | 0.008 | 0.993 |
| exclude_ambiguous_keys | building_held_out | normal_range_rule | 0.460 | 0.560 | 0.558 | 1.000 |
| exclude_ambiguous_keys | building_held_out | xgboost | 0.284 | 0.584 | 0.118 | 1.000 |
| exclude_ambiguous_keys | building_held_out | isolation_forest | 0.148 | 0.536 | 0.048 | 1.000 |
| exclude_ambiguous_keys | building_held_out | lstm_autoencoder | 0.138 | 0.523 | 0.051 | 0.993 |
| exclude_ambiguous_keys | chronological | xgboost | 0.823 | 0.909 | 0.176 | 1.000 |
| exclude_ambiguous_keys | chronological | normal_range_rule | 0.278 | 0.498 | 0.268 | 1.000 |
| exclude_ambiguous_keys | chronological | lstm_autoencoder | 0.091 | 0.489 | 0.078 | 0.987 |
| exclude_ambiguous_keys | chronological | isolation_forest | 0.034 | 0.500 | 0.017 | 1.000 |
| exclude_ambiguous_keys | nested_grouped | xgboost | 0.956 | 0.965 | 0.006 | 1.000 |
| exclude_ambiguous_keys | nested_grouped | normal_range_rule | 0.529 | 0.642 | 0.261 | 1.000 |
| exclude_ambiguous_keys | nested_grouped | isolation_forest | 0.087 | 0.519 | 0.010 | 1.000 |
| exclude_ambiguous_keys | nested_grouped | lstm_autoencoder | 0.026 | 0.503 | 0.008 | 0.994 |
