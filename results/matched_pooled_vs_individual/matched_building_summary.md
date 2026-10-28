# Matched pooled-versus-individual comparison

All strategies are evaluated on the same target-building test rows.
Individual models use target-building data only; pooled-adapted models add other buildings' training rows; pooled-transfer models do not see the target building during training.

| Building | Design | Strategy | Test F1 | Balanced accuracy | Recall | False-positive rate | Test rows |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| auditorium | building_ahu_held_out | individual_xgboost | 0.997 | 0.997 | 0.996 | 0.002 | 26,164 |
| auditorium | building_ahu_held_out | pooled_adapted_xgboost | 0.999 | 0.999 | 1.000 | 0.002 | 26,164 |
| auditorium | building_ahu_held_out | pooled_transfer_xgboost | 0.276 | 0.580 | 0.160 | 0.000 | 26,164 |
| auditorium | building_chronological | individual_xgboost | 1.000 | 1.000 | 1.000 | 0.000 | 22,685 |
| auditorium | building_chronological | pooled_adapted_xgboost | 0.996 | 0.996 | 0.992 | 0.000 | 22,685 |
| auditorium | building_chronological | pooled_transfer_xgboost | 0.275 | 0.580 | 0.159 | 0.000 | 22,685 |
| hospital | building_ahu_held_out | individual_xgboost | 0.077 | 0.507 | 0.041 | 0.028 | 16,420 |
| hospital | building_ahu_held_out | pooled_adapted_xgboost | 0.079 | 0.506 | 0.042 | 0.031 | 16,420 |
| hospital | building_ahu_held_out | pooled_transfer_xgboost | 0.070 | 0.498 | 0.038 | 0.041 | 16,420 |
| hospital | building_chronological | individual_xgboost | 0.880 | 0.958 | 0.959 | 0.043 | 13,136 |
| hospital | building_chronological | pooled_adapted_xgboost | 0.888 | 0.969 | 0.983 | 0.046 | 13,136 |
| hospital | building_chronological | pooled_transfer_xgboost | 0.261 | 0.575 | 0.152 | 0.003 | 13,136 |
| office | building_ahu_held_out | individual_xgboost | 0.997 | 0.997 | 0.995 | 0.000 | 35,102 |
| office | building_ahu_held_out | pooled_adapted_xgboost | 0.996 | 0.997 | 0.995 | 0.001 | 35,102 |
| office | building_ahu_held_out | pooled_transfer_xgboost | 0.666 | 0.765 | 0.991 | 0.461 | 35,102 |
| office | building_chronological | individual_xgboost | 0.996 | 0.997 | 0.995 | 0.001 | 35,114 |
| office | building_chronological | pooled_adapted_xgboost | 0.685 | 0.840 | 1.000 | 0.320 | 35,114 |
| office | building_chronological | pooled_transfer_xgboost | 0.608 | 0.775 | 1.000 | 0.449 | 35,114 |
