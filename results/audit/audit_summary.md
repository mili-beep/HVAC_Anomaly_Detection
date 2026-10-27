# Multibuilding AHU raw-data audit

Generated: `2026-10-27T16:08:28.027074+10:30`
Data root: `/Users/mili/Downloads/27147678`

> This is a descriptive audit. No cleaning, imputation, relabelling, or modelling was performed.

## File overview

| Building | Rows | Columns | AHUs | Timestamp success | SHA-256 (prefix) |
| --- | ---: | ---: | ---: | ---: | --- |
| auditorium | 113,375 | 18 | 13 | 100.00% | `acfcd58d1dff…` |
| hospital | 66,048 | 13 | 8 | 100.00% | `66feab7fe66d…` |
| office | 175,532 | 21 | 20 | 100.00% | `7dfe5660c6d5…` |

## Auditorium

- Source file: `auditorium_scientific_data.csv` (9,147,313 bytes)
- Timestamp range: `2022-01-01T00:00:00` to `2022-12-31T23:00:00`
- Timestamp parse failures: **0**
- AHU/timestamp duplicate rows: **0** across **0** keys
- Exact duplicate rows: **0**
- Chronological within-AHU order: **True**
- Forward gaps over one hour: **76**

### Labels

| Label | Rows |
| --- | ---: |
| Normal condition | 60,219 |
| Heating pump fault | 24,890 |
| Cooling pump fault | 19,292 |
| Supply fan fault | 6,525 |
| Return air temperature fault | 1,734 |
| Supply air temperature fault | 404 |
| Valve position fault | 311 |

Calendar coverage is measured against each AHU's inclusive observed date range. Across AHUs, **0** calendar days are absent from those inferred ranges.

### Notable field-quality counts

| Field | Missing | Placeholders | Non-finite |
| --- | ---: | ---: | ---: |
| `Heating supply temperature` | 57,395 | 0 | 0 |
| `Total heating pump` | 57,395 | 0 | 0 |
| `Heating pump 1` | 57,395 | 0 | 0 |
| `Heating pump 2` | 57,395 | 0 | 0 |
| `Heating pump 3` | 57,395 | 0 | 0 |
| `Cooling supply temperature 1` | 55,980 | 0 | 0 |
| `Cooling supply temperature 2` | 55,980 | 0 | 0 |
| `Total cooling pump` | 55,980 | 0 | 0 |
| `Cooling pump 1` | 55,980 | 0 | 0 |
| `Cooling pump 2` | 55,980 | 0 | 0 |

## Hospital

- Source file: `hosptial_scientific_data.csv` (5,621,617 bytes)
- Timestamp range: `2023-01-01T00:00:00` to `2023-12-21T23:00:00`
- Timestamp parse failures: **0**
- AHU/timestamp duplicate rows: **368** across **184** keys
- Exact duplicate rows: **0**
- Chronological within-AHU order: **False**
- Forward gaps over one hour: **48**

### Labels

| Label | Rows |
| --- | ---: |
| Normal condition | 53,757 |
| Return air temperature fault | 8,835 |
| Supply fan fault | 1,921 |
| Valve position fault | 1,525 |
| Supply air temperature fault | 9 |
| Cooling supply temperature fault | 1 |

Calendar coverage is measured against each AHU's inclusive observed date range. Across AHUs, **88** calendar days are absent from those inferred ranges.

### Notable field-quality counts

| Field | Missing | Placeholders | Non-finite |
| --- | ---: | ---: | ---: |
| `Cooling supply temperature` | 37,632 | 0 | 0 |
| `cooling pump` | 37,632 | 0 | 0 |
| `Heat Exchanger` | 28,416 | 0 | 0 |
| `Heating supply temperature` | 28,416 | 0 | 0 |
| `Return temperature` | 24 | 8,184 | 0 |
| `Supply air temperature` | 0 | 24 | 0 |

## Office

- Source file: `office_scientific_data.csv` (15,414,979 bytes)
- Timestamp range: `2023-12-01T00:00:00` to `2024-11-30T23:00:00`
- Timestamp parse failures: **0**
- AHU/timestamp duplicate rows: **0** across **0** keys
- Exact duplicate rows: **0**
- Chronological within-AHU order: **False**
- Forward gaps over one hour: **43**

### Labels

| Label | Rows |
| --- | ---: |
| Normal condition | 119,402 |
| Supply fan fault | 47,983 |
| Return air temperature fault | 6,999 |
| Supply air temperature fault | 1,031 |
| Valve position fault | 117 |

Calendar coverage is measured against each AHU's inclusive observed date range. Across AHUs, **0** calendar days are absent from those inferred ranges.

### Notable field-quality counts

| Field | Missing | Placeholders | Non-finite |
| --- | ---: | ---: | ---: |
| `Cooling supply temperature 1` | 102,120 | 0 | 0 |
| `Cooling supply temperature 2` | 102,120 | 0 | 0 |
| `Total cooling pump` | 102,120 | 0 | 0 |
| `cooling pump 1` | 102,120 | 0 | 0 |
| `cooling pump 2` | 102,120 | 0 | 0 |
| `cooling pump 3` | 102,120 | 0 | 0 |
| `cooling pump 4` | 102,120 | 0 | 0 |
| `Heating supply temperature1` | 73,412 | 0 | 0 |
| `Heating supply temperature2` | 73,412 | 0 | 0 |
| `Total heating pump` | 73,412 | 0 | 0 |
| `Heating pump 1` | 73,412 | 0 | 0 |
| `Heating pump 2` | 73,412 | 0 | 0 |
| `Heating pump 3` | 73,412 | 0 | 0 |
