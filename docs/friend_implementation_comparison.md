# Comparison with the Friend's AHU Implementation

## Purpose

This note compares the implementation described by the project team member with
the current IRPN multibuilding AHU implementation. It is a methodological
comparison, not a claim that either set of reported numbers is directly
interchangeable. The two implementations may use different feature lists,
fold definitions, quality policies and selection procedures. Therefore, the
reported scores are treated as hypotheses until they are reproduced under the
same frozen protocol.

The comparison is based on the supplied description of the friend's work and
the code, configuration, audit artefacts and saved results in this repository.

## Executive comparison

| Area | Friend's implementation | Current IRPN implementation | Assessment |
|---|---|---|---|
| Dataset | Real labelled AHU data from the three-building Park et al. (2025) collection | The same multibuilding AHU collection, mapped into one canonical schema | Comparable starting point |
| Data preparation | The description does not specify duplicate, placeholder or missing-value policies | Dedicated audit, canonical ingestion, duplicate-policy sensitivity, cleaning second audit and row-level quality flags | IRPN is stronger and more auditable |
| Initial building focus | Auditorium first because it has the broadest fault coverage and cleaner data | Auditorium was audited first, then the same ingestion and modelling design was extended to hospital and office | Both make a defensible staged choice |
| Unsupervised detection | One-Class SVM, reported at approximately 0.77 | Normal-range rule, Isolation Forest and LSTM autoencoder | The friend has a useful classical comparison; IRPN has a broader declared matrix |
| Supervised detection | XGBoost with class weighting and randomised search | XGBoost with class weighting and a fixed, pre-declared configuration; model selection uses validation data | Friend has stronger tuning breadth; IRPN has stronger reproducibility unless tuning is nested |
| Fault diagnosis | Exact fault performance is described mainly through cross-building transfer | Exact fault-type diagnosis with support filtering, per-fold metrics, unsupported-class accounting and confidence abstention | IRPN provides stronger diagnostic evidence |
| Leakage control | Explicit comparison of all columns versus a symptoms-only feature set | A default physical-signal feature set plus an operational-context sensitivity; no separate symptoms-only ablation yet | Friend currently has the clearer shortcut test; IRPN should add it |
| Evaluation split | AHU-held-out split is used to prevent row-level memorisation | Chronological, nested grouped, building-held-out and building-specific AHU-held-out assignments; pooled AHU-held-out is retained but disabled | IRPN tests more deployment conditions |
| Cross-building testing | Quick directional transfer results are reported | Formal building-held-out results and a matched individual/pooled/transfer comparison use identical target test rows | IRPN is more rigorous, but the friend’s directional framing is useful for communication |
| Data limitations | Uneven fault support and label provenance are acknowledged | These limitations are measured through audit summaries and fold-level unsupported-row counts | Both are appropriately cautious; IRPN documents them more explicitly |

## What the friend's implementation does well

### 1. It makes the central modelling question easy to understand

The friend separates two questions that are often accidentally mixed:

1. Can a model distinguish normal operation from a fault?
2. Can it identify the particular fault type?

That distinction is valuable for the report. A model can be excellent at the
first task and considerably weaker at the second. The current IRPN project also
supports these separate tasks, but the friend’s explanation is especially clear
for a non-specialist reader.

### 2. It explicitly tests shortcut dependence

The comparison between all columns (reported macro F1 of 0.986) and the
symptoms-only version (reported macro F1 of 0.876) is a strong idea. It asks
whether the model is learning general abnormal system behaviour or merely
reading an actuator/state value that nearly defines a label. This is more useful
than presenting one impressive score without examining what information produced
it.

The reported result also gives a useful operational interpretation: direct
actuator and equipment-status signals may be legitimate inputs in a real BMS,
but they make it difficult to claim that the model discovered a fault from its
indirect symptoms. Showing both versions is therefore better than silently
choosing whichever score is higher.

### 3. It recognises why a random row split is unsafe

Hourly observations from one AHU are highly correlated. Keeping rows from the
same AHU in both training and test sets can allow a model to recognise the unit
rather than generalise to an unseen unit. The friend’s AHU-held-out design is a
sound response to that risk and is consistent with IRPN’s use of grouped and
building-specific splits.

### 4. It gives a practical reason for choosing XGBoost

For tabular sensor data, XGBoost is a credible primary model because it is fast,
handles nonlinear interactions, works well with mixed-scale engineered
features, and exposes feature importance or other explanation aids. The
friend’s use of class weighting addresses the fact that normal rows greatly
outnumber fault rows.

### 5. It is honest about coverage limitations

The friend explicitly notes that some hospital fault types have very little
support and that pump faults occur only in the auditorium. It also notes that
the labels originate from one research team. Those caveats are essential: a
high score on a well-represented class is not evidence that the model can
diagnose every fault in a new building.

## What is stronger in the current IRPN implementation

### 1. A reproducible data-quality gate

IRPN audits file checksums, schemas, timestamps, duplicate keys, placeholders,
missingness, calendar coverage, label support and suspicious numeric values
before modelling. The hospital audit identified 184 duplicate AHU/timestamp
keys (368 rows), including conflicting labels. The project then froze two
explicit policies:

- `exclude_ambiguous_keys`: remove every row belonging to an ambiguous key;
- `retain_flagged_keys`: retain the rows but preserve a duplicate-quality flag.

The main cleaned table has 354,587 rows and zero duplicate keys; the retained
sensitivity table has 354,955 rows and keeps the flagged keys. Neither policy
silently selects the first duplicate or averages conflicting observations.
This makes the later model comparison substantially easier to defend.

### 2. More than one notion of generalisation

IRPN does not rely on one AHU-held-out result. It has saved assignments for:

- chronological testing, which approximates future data;
- nested grouped testing, which reduces unit/session memorisation;
- building-held-out testing, which tests transfer to a building not used for
  training; and
- building-specific AHU-held-out testing.

The pooled `ahu_held_out` assignment is preserved for reproducibility but is
currently disabled from headline pooled comparisons. This avoids presenting
many similar folds as if they represented independent deployment scenarios.

### 3. Fold-local preprocessing and gap-aware temporal features

Imputation and scaling are fitted on training data within each fold. Rolling
features and rates reset at timestamp gaps, and LSTM sequences do not bridge an
outage or fabricate missing rows. These details reduce leakage and make the
temporal features more defensible than a simple global interpolation.

### 4. Formal matched pooled-versus-individual comparison

The new matched comparison evaluates an individual building model, a pooled
adapted model and a zero-shot transfer model on exactly the same target-building
test rows. This is stronger than comparing scores produced by unrelated runs.
It also shows that the answer is not universally “one model for everything”:
individual and pooled-adapted models are close in some settings, while transfer
without target-building adaptation is substantially weaker.

### 5. More transparent diagnosis reporting

IRPN records macro and weighted metrics, class support, unsupported rows when a
class is absent from training, and a 70% confidence rule that can abstain. This
prevents a strong weighted average from hiding poor performance on rare fault
types.

## Limitations of the friend's implementation

These are not necessarily coding errors. They are items that must be clarified
before its results can be compared with the IRPN headline results.

### 1. The reported numbers are described as early results

The cross-building figures (approximately 88% hospital recall, 99.9% office
recall and 0.3–0.4 macro F1 for exact fault transfer) are explicitly described
as a quick run. Without the exact folds, row counts, duplicate policy,
feature manifest and confidence intervals, they should not be treated as final
evidence.

### 2. Randomised search needs a strict validation boundary

Randomised hyperparameter search is reasonable, but it must search only within
training/validation data. If the test set influenced the search, feature
selection or threshold choice, the score is optimistic. IRPN currently uses a
fixed XGBoost configuration, which is less broad but easier to reproduce. A
tuned variant should be added only as a separately declared sensitivity run,
preferably with an inner validation loop.

### 3. “Symptoms-only” is not yet a fully specified feature policy

The supplied description says symptoms-only performance is about 0.876, but it
does not define every included and excluded column. A fair comparison needs a
versioned feature manifest. Otherwise, it is impossible to know whether the
change came from removing direct commands, removing all actuator signals,
removing temporal features, or simply using temperature columns.

### 4. AHU-held-out testing is not the same as future-building testing

An unseen AHU split answers whether the model transfers to another unit under
the same building and data-generating process. It does not answer whether a
model trained on the auditorium transfers to an office or hospital. The friend
mentions directional transfer, which is valuable, but the protocol must be
separated from the within-building split.

### 5. Missingness and duplicate handling are not documented

The description does not say whether hospital duplicate keys were removed,
retained, averaged or silently deduplicated, nor whether missing values were
imputed globally or within folds. Those decisions can materially affect both
the model and the apparent generalisation result.

### 6. Exact fault-type support is uneven

The hospital’s under-represented classes and the auditorium-only pump faults
mean that a global fault classifier cannot be interpreted as equally capable of
diagnosing every fault. Per-building and per-fault support must accompany every
diagnosis score.

## Recommendations for strengthening IRPN

### Adopt now: explicit shortcut/symptom sensitivity

Add one declared XGBoost comparison using the same cleaned rows, folds,
preprocessing and test assignments as the headline model:

| Variant | Purpose |
|---|---|
| `default_bms_signals` | The practical model using the approved canonical sensors, physical relationships and past-only temporal features |
| `symptom_only` | A stricter scientific sensitivity that excludes direct command/status channels likely to reveal the fault setting |

The feature manifest must list every included and excluded column. Report the
feature count, fold-level macro F1, balanced accuracy, recall, false-positive
rate, test-row count and the largest feature importances. The symptoms-only
variant should not replace the practical model silently; it should reveal how
much performance depends on direct equipment-state information.

This is the most valuable recommendation from the friend's implementation.

### Adopt now: clearer transfer reporting

Keep the existing formal building-held-out and matched comparisons, but add a
small directional table when the protocol is useful:

- auditorium training -> office testing;
- office training -> auditorium testing;
- auditorium + office training -> hospital testing.

State whether the source-building data are adapted with target-building labels
or whether this is zero-shot transfer. Do not combine these values into one
average, because they answer different operational questions.

### Adopt now: support and coverage tables for diagnosis

For every building and split, report training, validation and test counts for
each fault type. Mark a result as unsupported when the class is absent from the
training fold. Use macro F1 as the primary diagnosis summary and weighted F1 as
a secondary context measure. Keep the confidence abstention output so staff
are not given an exact diagnosis for a class the model has not learned
reliably.

### Consider as a controlled sensitivity: tuned XGBoost

Implement a separate `xgboost_tuned` configuration with a pre-declared search
space and a validation-only or nested search. Preserve the current fixed
configuration as the reproducible reference. Adopt the tuned model only if it
improves the same held-out folds without increasing false alarms or reducing
coverage.

### Optional, lower priority: One-Class SVM

The friend's One-Class SVM is a reasonable classical unsupervised comparator,
but it should not displace the current models. Add it only if the report needs
a broader unsupervised comparison and the runtime is acceptable. It cannot
provide fault-type diagnosis and should not be used to justify the supervised
model selection.

### Do not adopt without evidence

- Do not remove columns solely because they improve or reduce a score after the
  result is known.
- Do not report the friend’s early transfer numbers as if they were an IRPN
  experiment.
- Do not pool rare fault types into a single diagnosis score without showing
  the original class support.
- Do not use a random row split as the headline evaluation.
- Do not tune thresholds, imputation or feature lists using the final test
  rows.

## Implementation plan

### Step 1 — Freeze the comparison protocol

Document the existing duplicate policy, canonical feature table, active splits,
target labels and primary metrics. Freeze the current XGBoost configuration and
record the exact test-row identities. No new feature exclusion or threshold is
selected from the results of the new runs.

**Acceptance:** a manifest identifies the data policy, feature manifest,
split file, model configuration and output directory for every comparison.

### Step 2 — Define the two feature manifests

Create explicit lists for `default_bms_signals` and `symptom_only`. The second
list should be agreed before training and should describe the scientific
question, not merely remove the columns that appear important. Keep metadata,
labels, missingness flags and building identifiers out of both model inputs.

**Acceptance:** a test verifies that the two lists are disjoint from metadata
and that every predictor is present in the feature table.

### Step 3 — Run the shortcut/symptom ablation

Use the same XGBoost implementation, fold assignments and fold-local
preprocessing for both variants. Run the main duplicate policy first and retain
the duplicate policy as a separate sensitivity output. Save fold-level metrics,
row counts, confusion matrices, feature importances and a feature manifest.

**Acceptance:** the two variants are evaluated on identical test rows and can
be compared in one table without mixing folds or policies.

### Step 4 — Add tuned XGBoost as a secondary experiment

Define a small search space for tree count, depth, learning rate, subsampling
and regularisation. Search only on training/validation rows. If a nested
grouped design is affordable, use it; otherwise use the pre-declared validation
fold and label the result accordingly. Do not tune on the final test rows.

**Acceptance:** the selected parameters and search seed are saved, and the
fixed-config headline result remains unchanged.

### Step 5 — Add directional transfer summaries

Generate auditorium-to-office, office-to-auditorium and auditorium/office-to-
hospital summaries using the existing transfer code. Clearly label zero-shot
transfer versus target-adapted pooled training. Include test support, macro F1,
weighted F1, recall and false-positive rate.

**Acceptance:** each direction has a declared source, target, training rows,
test rows and no overlap between source training and target test identifiers.

### Step 6 — Strengthen diagnosis evidence

Generate per-building and per-fault support tables for the existing exact
fault-type experiment. Add an explicit “not evaluated” or “unsupported” label
when a class is absent from the training fold. Keep confidence abstention and
show coverage alongside macro and weighted scores.

**Acceptance:** no `NaN` diagnosis score is presented without an explanation of
the missing class support or evaluation coverage.

### Step 7 — Update the notebooks and report

Add one clear secondary section to the pooled detection notebook:

> Shortcut/symptom sensitivity: practical BMS signals versus symptoms-only.

Add the directional transfer table and support/coverage figures to the relevant
results notebook. Keep the main story short: one practical model, one
scientific sensitivity, and clearly separated transfer evidence.

**Acceptance:** a reader can identify the headline model, the leakage/shortcut
check, and the limits of cross-building diagnosis without reading the code.

### Step 8 — Decision gate

Choose the final headline configuration using pre-declared criteria:

- no test-row leakage;
- acceptable false-positive rate;
- stable performance across the intended split;
- sufficient diagnosis coverage;
- a feature set that is operationally available and scientifically defensible.

The practical default may remain the pooled-adapted XGBoost model for
simplicity if its matched results are close to individual models. The
symptoms-only model should remain as a robustness and shortcut-sensitivity
comparison, not be selected merely because it gives a preferred narrative.

## Recommended final position

The friend's implementation contributes a particularly strong idea: make the
effect of direct equipment-state signals visible instead of hiding it. IRPN is
already stronger in data audit, fold discipline, quality-control policies,
cross-building evaluation and diagnosis accounting. The best next step is
therefore not to replace IRPN with the friend's code. It is to reproduce the
friend's shortcut/symptom comparison inside the existing, auditable IRPN
pipeline, then add a carefully controlled tuned-XGBoost sensitivity only if
the search can be kept separate from final test evaluation.

This gives the project a defensible narrative: the practical model uses the
signals a BMS would normally expose, while the stricter symptoms-only result
shows how much of the performance remains when direct fault-setting channels
are withheld.

