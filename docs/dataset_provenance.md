# Dataset provenance

## Source

The project uses the real multibuilding AHU data associated with the Scientific
Data article by Wang et al. and Figshare record 27147678 (version 3):

- Article DOI: <https://doi.org/10.1038/s41597-025-05825-9>
- Data DOI: <https://doi.org/10.6084/m9.figshare.27147678.v3>

The source files describe operational BMS data from auditorium, hospital, and
office buildings. They are not copied into this repository because the raw
files are large and their redistribution/licensing terms must be respected.

## Local snapshot recorded

The following checksums were recorded from the local snapshot at
`/Users/mili/Downloads/27147678/` on 2026-10-27. Recompute them after any
download or file replacement:

| File | SHA-256 |
| --- | --- |
| `auditorium_scientific_data.csv` | `acfcd58d1dff5b76f64bbe716e824c4592f83a5cbfc7e6c8b77fe050bd414d3b` |
| `hosptial_scientific_data.csv` | `66feab7fe66d77ca1c632315a61a7a93412ff00dc2fb0cd9b2485394cecca9f7` |
| `office_scientific_data.csv` | `7dfe5660c6d5b0db388e01a59956a3ce40136d42c7a1d2038bfc6a7ea2a80203` |

The supplied processing notebook is recorded for reference only and is not
treated as the project implementation:

`FDD_processing.ipynb` —
`c0788f09fbc460a512bdd2d1e5e87845d3341e99f26457c4805109387224cef7`.

## Initial audit concerns

These are hypotheses to verify in Phase 1, not modelling decisions:

- hospital rows include duplicate AHU/timestamp keys and conflicting labels;
- literal `.` placeholders occur in hospital sensor fields;
- missingness is partly determined by operating mode and subsystem activity;
- some fault classes have very low support;
- label distributions are highly uneven across AHUs and buildings;
- an AHU identifier must not be used as a predictive feature;
- cross-building performance may differ substantially from within-building
  performance.

The audit will decide whether each issue is repaired, excluded, retained with a
quality flag, or reported only as a limitation case study.
