# Building-specific detection results

Models were selected using validation F1, with balanced accuracy as a tie-break. The selected model was then retrained on training plus validation rows and evaluated once on the held-out test rows.

## Selected models and test results

| Building | Design | Selected model | Validation F1 | Test F1 | Test balanced accuracy | Test FPR | Test coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| auditorium | building_ahu_held_out | xgboost | 0.999 | 0.997 | 0.997 | 0.002 | 1.000 |
| auditorium | building_chronological | xgboost | 0.996 | 1.000 | 1.000 | 0.000 | 1.000 |
| hospital | building_ahu_held_out | xgboost | 0.789 | 0.077 | 0.507 | 0.028 | 1.000 |
| hospital | building_chronological | xgboost | 0.997 | 0.880 | 0.958 | 0.043 | 1.000 |
| office | building_ahu_held_out | xgboost | 0.998 | 0.997 | 0.997 | 0.000 | 1.000 |
| office | building_chronological | xgboost | 0.997 | 0.996 | 0.997 | 0.001 | 1.000 |

## Validation comparison

See `building_detection_validation_metrics.csv` for every model.
