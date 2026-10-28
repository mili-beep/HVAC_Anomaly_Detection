# Binary baseline comparison

These are fold-level averages. The statistical and ML models must use the same saved assignments.

| Design | Baseline | Mean F1 | Mean balanced accuracy | Mean false-positive rate |
| --- | --- | ---: | ---: | ---: |
| building_held_out | normal_range_rule | 0.429 | 0.581 | 0.346 |
| building_held_out | actuator_pump_only | 0.029 | 0.502 | 0.012 |
| building_held_out | majority_class | 0.000 | 0.500 | 0.000 |
| chronological | actuator_pump_only | 0.398 | 0.617 | 0.037 |
| chronological | normal_range_rule | 0.178 | 0.510 | 0.101 |
| chronological | majority_class | 0.000 | 0.500 | 0.000 |
| nested_grouped | actuator_pump_only | 0.492 | 0.648 | 0.094 |
| nested_grouped | normal_range_rule | 0.448 | 0.628 | 0.080 |
| nested_grouped | majority_class | 0.000 | 0.500 | 0.000 |
