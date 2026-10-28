# Shared multibuilding feature engineering

Step 5 uses one feature-generation implementation for auditorium, hospital and
office. It consumes the quality-controlled main table from Step 4 and does not
fit an imputer or use labels to create predictors.

The default physical-signal variant contains canonical sensor values,
temperature/setpoint and actuator/pump relationships, plus past-only rolling
means and rates of change. The rolling functions group by building and AHU, so
history cannot cross buildings or AHUs. A rate is only calculated when the
previous observation is positive and within the configured two-hour limit.

The operational-context variant contains the same default predictors plus
missingness/availability indicators, derived heating/cooling operating-mode
indicators and cyclical hour/weekday context. It is a secondary XGBoost
sensitivity comparison, not the headline feature set. It is intentionally
kept separate because availability and time context can reflect logging or
operating practice rather than physical fault behaviour.

Labels are carried only as evaluation keys. A separate shortcut audit reports
label associations; it does not remove features or change the declared model
inputs. Imputation remains a later, fold-specific modelling operation.

Run the generator with:

    engineer_multibuilding_features \\
      --input-file data/interim/cleaned/cleaned_ahu.csv \\
      --output-dir data/interim/features \\
      --building-config configs/multibuilding_ahu.yaml \\
      --feature-config configs/feature_engineering.yaml

The output directory contains one feature table, a lineage table, a shortcut
audit, a manifest and a concise summary. The manifest lists the columns for
both variants, so downstream experiments select columns without maintaining a
second feature implementation or duplicating the large feature table.

