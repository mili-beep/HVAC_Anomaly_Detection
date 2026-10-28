# Single-feature held-out dependence audit

Each feature was evaluated by itself on held-out rows. Missing values were filled with the training-fold median only.
A direction-free AUC close to 0.5 means little independent separation; a value close to 1 means a strong one-feature shortcut.

| Feature | Family | Valid folds | Mean absolute AUC | Max absolute AUC |
| --- | --- | ---: | ---: | ---: |
| temporal__past_mean_24__return_temperature | past_rolling_mean | 9/9 | 0.667 | 0.816 |
| sensor__return_temperature | raw_sensor | 9/9 | 0.679 | 0.812 |
| temporal__past_mean_3__return_temperature | past_rolling_mean | 9/9 | 0.673 | 0.810 |
| relationship__return_setpoint_residual | relationship | 9/9 | 0.645 | 0.803 |
| relationship__heating_pump_any_on | pump_relationship | 8/9 | 0.647 | 0.787 |
| relationship__supply_return_abs_delta | relationship | 9/9 | 0.589 | 0.746 |
| temporal__past_mean_24__setpoint_temperature | past_rolling_mean | 9/9 | 0.597 | 0.711 |
| sensor__setpoint_temperature | raw_sensor | 9/9 | 0.594 | 0.707 |
| temporal__past_mean_3__setpoint_temperature | past_rolling_mean | 9/9 | 0.594 | 0.707 |
| sensor__heating_pump_total | raw_sensor | 7/9 | 0.640 | 0.689 |
| temporal__past_mean_3__heating_pump_total | past_rolling_mean | 7/9 | 0.623 | 0.681 |
| relationship__supply_return_delta | relationship | 9/9 | 0.598 | 0.679 |
| temporal__past_mean_24__heating_pump_total | past_rolling_mean | 7/9 | 0.614 | 0.678 |
| sensor__cooling_supply_temperature_1 | raw_sensor | 9/9 | 0.563 | 0.667 |
| relationship__supply_setpoint_residual | relationship | 9/9 | 0.581 | 0.665 |
| temporal__past_mean_24__heating_pump_3 | past_rolling_mean | 7/9 | 0.595 | 0.665 |
| temporal__past_mean_3__cooling_supply_temperature_1 | past_rolling_mean | 9/9 | 0.560 | 0.663 |
| temporal__past_mean_24__cooling_pump_total | past_rolling_mean | 9/9 | 0.572 | 0.662 |
| temporal__past_mean_24__cooling_supply_temperature_1 | past_rolling_mean | 9/9 | 0.561 | 0.661 |
| sensor__heating_pump_3 | raw_sensor | 7/9 | 0.599 | 0.661 |
