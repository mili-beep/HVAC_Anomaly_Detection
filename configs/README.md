# Configuration

`auditorium.yaml` is the single frozen policy for the auditorium source,
feature roles, cleaning rules, feature engineering and normal-operation
pattern discovery. Keeping these related declarations together prevents the
same sensor list from drifting across phase-specific files.

`multibuilding_ahu.yaml` remains separate because it describes source-to-
canonical mappings for all three buildings rather than the auditorium policy.
Future building policies should follow the same one-file-per-building pattern.
