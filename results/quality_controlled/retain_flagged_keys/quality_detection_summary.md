# Quality-controlled pooled detection summary

All models use the declared physical-signal feature variant. Imputation and scaling are fitted within each training fold.

| Policy | Design | Model | Mean F1 | Mean balanced accuracy | Mean FPR | Mean coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| retain_flagged_keys | building_held_out | normal_range_rule | 0.460 | 0.560 | 0.558 | 1.000 |
| retain_flagged_keys | building_held_out | xgboost | 0.267 | 0.570 | 0.116 | 1.000 |
| retain_flagged_keys | building_held_out | isolation_forest | 0.164 | 0.541 | 0.072 | 1.000 |
| retain_flagged_keys | building_held_out | lstm_autoencoder | 0.131 | 0.520 | 0.052 | 0.991 |
| retain_flagged_keys | chronological | xgboost | 0.820 | 0.907 | 0.180 | 1.000 |
| retain_flagged_keys | chronological | normal_range_rule | 0.278 | 0.498 | 0.268 | 1.000 |
| retain_flagged_keys | chronological | lstm_autoencoder | 0.088 | 0.488 | 0.078 | 0.987 |
| retain_flagged_keys | chronological | isolation_forest | 0.047 | 0.504 | 0.018 | 1.000 |
| retain_flagged_keys | nested_grouped | xgboost | 0.955 | 0.964 | 0.007 | 1.000 |
| retain_flagged_keys | nested_grouped | normal_range_rule | 0.529 | 0.642 | 0.261 | 1.000 |
| retain_flagged_keys | nested_grouped | isolation_forest | 0.087 | 0.519 | 0.009 | 1.000 |
| retain_flagged_keys | nested_grouped | lstm_autoencoder | 0.027 | 0.503 | 0.008 | 0.993 |
