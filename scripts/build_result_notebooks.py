"""Build the four presentation notebooks from the saved project artefacts."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def save(name: str, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, NOTEBOOK_DIR / name)


COMMON = """
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'normal': '#457b9d', 'fault': '#e76f51', 'accent': '#2a9d8f', 'dark': '#264653'}

def find_root():
    for candidate in [Path.cwd(), Path.cwd().parent, Path('/Users/mili/Desktop/IRPN')]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'results').exists():
            return candidate
    raise FileNotFoundError('Open this notebook from the IRPN repository or set ROOT manually.')

ROOT = find_root()

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def require(*paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing saved artefacts. Run the pipeline commands first:\\n' + '\\n'.join(missing))

def pct(value):
    return f'{100 * value:.1f}%' if pd.notna(value) else 'n/a'
"""


def notebook_01() -> None:
    cells = [
        markdown(
            """
            # 01 — Audit, cleaning, features and normal patterns

            This notebook is the evidence walkthrough for the data-preparation stages. It does not train a model. It loads the saved audit, cleaning, feature-engineering and normal-operation artefacts so that the decisions can be inspected before modelling.

            The headline data policy is the quality-controlled table with ambiguous duplicate keys excluded. The retained-duplicate table is shown only as a declared sensitivity comparison. Normal-operation patterns are descriptive and are not presented as proof of generalisation.
            """
        ),
        code(COMMON),
        markdown("## 1. Load the saved artefacts\n\nThe paths below are repository-relative and are checked before any figures are created."),
        code(
            """
AUDIT = ROOT / 'results' / 'audit'
CLEANED = ROOT / 'data' / 'interim' / 'cleaned'
FEATURES = ROOT / 'data' / 'interim' / 'features'
PATTERNS = ROOT / 'data' / 'processed' / 'auditorium' / 'patterns'
required = [
    AUDIT / 'audit_manifest.json', AUDIT / 'audit_label_counts.csv', AUDIT / 'audit_missingness.csv',
    CLEANED / 'cleaning_verification_manifest.json', CLEANED / 'cleaning_second_audit.csv',
    FEATURES / 'feature_manifest.json', FEATURES / 'feature_lineage.csv', FEATURES / 'shortcut_audit.csv',
    PATTERNS / 'pattern_manifest.json', PATTERNS / 'hourly_profile.csv', PATTERNS / 'pca_coordinates.csv',
]
require(*required)
audit_manifest = read_json(AUDIT / 'audit_manifest.json')
cleaning_manifest = read_json(CLEANED / 'cleaning_verification_manifest.json')
feature_manifest = read_json(FEATURES / 'feature_manifest.json')
pattern_manifest = read_json(PATTERNS / 'pattern_manifest.json')
print('Repository:', ROOT)
print('Audit generated:', audit_manifest.get('generated_at'))
print('Quality-controlled rows:', cleaning_manifest.get('main', {}).get('row_count', 'see manifest'))
print('Physical features:', len(feature_manifest['feature_variants']['physical_signal']))
print('Operational-context features:', len(feature_manifest['feature_variants']['operational_context']))
"""
        ),
        markdown("## 2. Dataset audit and cleaning decision\n\nThe audit is deliberately shown before the model results. It records the data that was available, while the second audit records the effect of the frozen cleaning policy."),
        code(
            """
label_counts = pd.read_csv(AUDIT / 'audit_label_counts.csv')
missingness = pd.read_csv(AUDIT / 'audit_missingness.csv')
second_audit = pd.read_csv(CLEANED / 'cleaning_second_audit.csv')
display(label_counts.head(20))
display(second_audit.head(20))
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
labels_plot = label_counts.copy()
if 'label' in labels_plot.columns:
    counts = labels_plot.groupby('label', dropna=False)['count'].sum().sort_values()
    axes[0].barh(counts.index.astype(str), counts.values, color=[COLORS['normal'] if 'Normal' in str(x) else COLORS['fault'] for x in counts.index])
    axes[0].set_title('Source label support')
    axes[0].set_xlabel('Rows')
miss_col = 'missing_count' if 'missing_count' in missingness.columns else missingness.columns[-1]
feature_col = 'feature' if 'feature' in missingness.columns else missingness.columns[0]
top_missing = missingness.sort_values(miss_col, ascending=False).head(12)
axes[1].barh(top_missing[feature_col].astype(str)[::-1], top_missing[miss_col][::-1], color=COLORS['accent'])
axes[1].set_title('Highest missing-value counts')
axes[1].set_xlabel('Rows')
plt.show()
"""
        ),
        markdown(
            """
            **Interpretation.** Missing heating/cooling values are treated as potentially structural operating-mode availability, not automatically as faults. The main policy excludes every row belonging to an ambiguous `building + AHU + timestamp` key; it does not keep an arbitrary duplicate or average conflicting observations. No global imputation is performed.
            """
        ),
        markdown("## 3. Feature variants and shortcut review\n\nThe physical-signal variant is the default modelling input. The operational-context variant is retained only as a sensitivity comparison."),
        code(
            """
variant_table = pd.DataFrame({
    'variant': ['physical_signal', 'operational_context'],
    'feature_count': [len(feature_manifest['feature_variants']['physical_signal']), len(feature_manifest['feature_variants']['operational_context'])],
    'description': [
        'Sensors, relationships and past-only temporal features',
        'Physical features plus missingness, modes and cyclical time context',
    ],
})
display(variant_table)
lineage = pd.read_csv(FEATURES / 'feature_lineage.csv')
shortcut = pd.read_csv(FEATURES / 'shortcut_audit.csv')
display(lineage.groupby(['family', 'variant'], dropna=False).size().rename('features').reset_index())
display(shortcut.sort_values('abs_label_mean_difference', ascending=False).head(15))
"""
        ),
        markdown(
            """
            The shortcut audit is diagnostic only. It helps identify suspicious univariate relationships, but it does not remove a feature after looking at model performance. Feature selection is declared by the manifest and is independent of the test labels.
            """
        ),
        markdown("## 4. Normal-operation patterns",),
        code(
            """
hourly = pd.read_csv(PATTERNS / 'hourly_profile.csv')
pca = pd.read_csv(PATTERNS / 'pca_coordinates.csv')
display(hourly.head(12))
display(pd.DataFrame([pattern_manifest]).T.rename(columns={0: 'value'}).head(20))
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
if {'hour', 'mean'}.issubset(hourly.columns):
    for sensor, group in hourly.groupby('sensor'):
        if sensor in set(hourly['sensor'].drop_duplicates().head(4)):
            axes[0].plot(group['hour'], group['mean'], marker='o', label=str(sensor))
    axes[0].set_title('Normal-operation hourly profiles')
    axes[0].set_xlabel('Hour of day')
    axes[0].set_ylabel('Mean sensor value')
    axes[0].legend(fontsize=8)
if {'pc1', 'pc2'}.issubset(pca.columns):
    colour = pca['mode'].astype(str) if 'mode' in pca.columns else COLORS['accent']
    axes[1].scatter(pca['pc1'], pca['pc2'], c=pd.factorize(colour)[0], cmap='viridis', alpha=0.55, s=18)
    axes[1].set_title('PCA of normal operating observations')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
plt.show()
"""
        ),
        markdown(
            """
            **Boundary of this evidence.** The profiles describe the normal rows available in the auditorium data. They support feature interpretation and visual inspection; they do not establish that the same patterns hold in another building or season.
            """
        ),
        markdown("## 5. Preparation acceptance summary"),
        code(
            """
summary = pd.DataFrame([
    ['Canonical input', audit_manifest.get('row_count', np.nan), 'Raw mapped rows before quality policy'],
    ['Main cleaned table', cleaning_manifest.get('main', {}).get('row_count', np.nan), 'Ambiguous duplicate keys excluded'],
    ['Retained sensitivity table', cleaning_manifest.get('sensitivity', {}).get('row_count', np.nan), 'Duplicate keys retained and flagged'],
    ['Default features', len(feature_manifest['feature_variants']['physical_signal']), 'Used for headline models'],
    ['Context sensitivity features', len(feature_manifest['feature_variants']['operational_context']), 'Used only for comparison'],
], columns=['artefact', 'value', 'meaning'])
display(summary)
"""
        ),
    ]
    save("01_auditorium_pipeline_walkthrough.ipynb", cells)


def notebook_02() -> None:
    cells = [
        markdown(
            """
            # 02 — Detection experiments and generalisation

            This notebook presents the binary normal-versus-fault detection evidence. It does not retrain models. It compares the earlier pooled results with the quality-controlled rerun, shows the duplicate-policy sensitivity, and then reports the separate auditorium, hospital and office models.

            The default headline feature track is the physical-signal variant. Operational-context features appear only once as a secondary sensitivity comparison in the earlier pooled results.
            """
        ),
        code(COMMON),
        markdown("## 1. Load saved detection artefacts"),
        code(
            """
BASELINE = ROOT / 'results' / 'baselines'
OLD = ROOT / 'results' / 'detection'
QUALITY_MAIN = ROOT / 'results' / 'quality_controlled' / 'exclude_ambiguous_keys'
QUALITY_RETAIN = ROOT / 'results' / 'quality_controlled' / 'retain_flagged_keys'
BUILDING = ROOT / 'results' / 'building_specific'
require(
    BASELINE / 'baseline_metrics_summary.csv', OLD / 'detection_metrics_summary.csv',
    QUALITY_MAIN / 'quality_detection_metrics_by_fold.csv', QUALITY_RETAIN / 'quality_detection_metrics_by_fold.csv',
    BUILDING / 'building_detection_validation_metrics.csv', BUILDING / 'building_detection_test_metrics.csv',
)
baseline = pd.read_csv(BASELINE / 'baseline_metrics_summary.csv')
old_detection = pd.read_csv(OLD / 'detection_metrics_summary.csv')
quality_main = pd.read_csv(QUALITY_MAIN / 'quality_detection_metrics_by_fold.csv')
quality_retain = pd.read_csv(QUALITY_RETAIN / 'quality_detection_metrics_by_fold.csv')
building_validation = pd.read_csv(BUILDING / 'building_detection_validation_metrics.csv')
building_test = pd.read_csv(BUILDING / 'building_detection_test_metrics.csv')
print('Earlier pooled rows:', len(old_detection))
print('Quality-controlled fold rows:', len(quality_main))
print('Building-specific validation rows:', len(building_validation))
"""
        ),
        markdown("## 2. Baseline comparison"),
        code(
            """
display(baseline.round(3))
"""
        ),
        code(
            """
baseline_plot = baseline.sort_values('mean_f1')
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
ax.barh(baseline_plot['baseline'], baseline_plot['mean_f1'], color=COLORS['accent'])
ax.set_xlim(0, 1)
ax.set_xlabel('Mean F1')
ax.set_title('Transparent binary baselines')
for index, value in enumerate(baseline_plot['mean_f1']):
    ax.text(value + 0.01, index, f'{value:.3f}', va='center')
plt.show()
"""
        ),
        markdown(
            """
            The baselines are references, not competing deployment products. The normal-range rule is transparent and interpretable; the majority class is a lower bound; the actuator/pump-only baseline tests whether a small control subset is sufficient.
            """
        ),
        markdown("## 3. Earlier pooled XGBoost versus the quality-controlled rerun"),
        code(
            """
old_xgb = old_detection[(old_detection['model'] == 'xgboost') & (old_detection['variant'] == 'physical_signal')].copy()
old_xgb = old_xgb[['design', 'mean_f1', 'mean_balanced_accuracy', 'mean_false_positive_rate']].rename(columns={
    'mean_f1': 'earlier_f1', 'mean_balanced_accuracy': 'earlier_balanced_accuracy', 'mean_false_positive_rate': 'earlier_fpr'
})
new_xgb = quality_main[quality_main['model'] == 'xgboost'].groupby('design', as_index=False).agg(
    quality_f1=('f1', 'mean'), quality_balanced_accuracy=('balanced_accuracy', 'mean'), quality_fpr=('false_positive_rate', 'mean')
)
comparison = old_xgb.merge(new_xgb, on='design', how='outer')
comparison['f1_change'] = comparison['quality_f1'] - comparison['earlier_f1']
display(comparison.round(3))
"""
        ),
        code(
            """
plot_data = comparison.sort_values('earlier_f1')
positions = np.arange(len(plot_data))
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.barh(positions - 0.18, plot_data['earlier_f1'], height=0.35, label='Earlier pooled run', color='#8ecae6')
ax.barh(positions + 0.18, plot_data['quality_f1'], height=0.35, label='Quality-controlled run', color=COLORS['dark'])
ax.set_yticks(positions, plot_data['design'])
ax.set_xlim(0, 1)
ax.set_xlabel('Mean F1')
ax.set_title('Physical-signal XGBoost: protocol comparison')
ax.legend()
plt.show()
"""
        ),
        markdown(
            """
            This comparison is not a claim that the two runs are identical. The new run uses cleaned data, 88 declared physical predictors, duplicate-aware temporal processing and the quality-controlled split table. It is therefore the more defensible headline result. The main change is the harder hospital building-held-out fold.
            """
        ),
        markdown("## 4. Duplicate-policy sensitivity"),
        code(
            """
def xgb_summary(frame, policy):
    return frame[frame['model'].eq('xgboost')].groupby('design', as_index=False).agg(
        policy=('duplicate_policy', 'first'), mean_f1=('f1', 'mean'), mean_fpr=('false_positive_rate', 'mean')
    )
duplicate_comparison = pd.concat([xgb_summary(quality_main, 'exclude'), xgb_summary(quality_retain, 'retain')], ignore_index=True)
display(duplicate_comparison.round(3))
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
for policy, group in duplicate_comparison.groupby('policy'):
    ax.plot(group['design'], group['mean_f1'], marker='o', linewidth=2, label=policy)
ax.set_ylim(0, 1)
ax.set_ylabel('Mean F1')
ax.set_title('XGBoost sensitivity to duplicate policy')
ax.tick_params(axis='x', rotation=20)
ax.legend(title='Policy')
plt.show()
"""
        ),
        markdown("## 5. Independent building models"),
        code(
            """
display(building_test[['building', 'design', 'model', 'f1', 'balanced_accuracy', 'false_positive_rate', 'coverage']].round(3))
"""
        ),
        code(
            """
building_plot = building_test.copy()
building_plot['label'] = building_plot['building'] + ' / ' + building_plot['design'].str.replace('building_', '', regex=False)
building_plot = building_plot.sort_values('f1')
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.barh(building_plot['label'], building_plot['f1'], color=[COLORS['fault'] if value < 0.7 else COLORS['dark'] for value in building_plot['f1']])
ax.set_xlim(0, 1)
ax.set_xlabel('Test F1 of selected model')
ax.set_title('Building-specific held-out test results')
for index, value in enumerate(building_plot['f1']):
    ax.text(value + 0.01, index, f'{value:.3f}', va='center')
plt.show()
"""
        ),
        markdown(
            """
            **Interpretation.** XGBoost was selected for all six building/design combinations. Auditorium and office are highly consistent. Hospital chronological performance is lower but usable as a research result. Hospital AHU-held-out performance is poor, showing that a model trained on some hospital AHUs does not automatically generalise to unseen hospital AHUs.
            """
        ),
        markdown("## 6. What the experiments do and do not establish"),
        code(
            """
conclusions = pd.DataFrame([
    ['Within-building grouped evaluation', 'Strong XGBoost results', 'Useful evidence for the labelled buildings and split design'],
    ['Chronological evaluation', 'Strong for auditorium/office; lower for hospital', 'Temporal drift and building-specific calibration matter'],
    ['Unseen-building evaluation', 'Much weaker, especially hospital', 'Cross-building deployment cannot be assumed'],
    ['Duplicate sensitivity', 'Small change between policies', 'Headline pooled result is not driven by duplicate rows'],
], columns=['question', 'observed result', 'responsible interpretation'])
display(conclusions)
"""
        ),
    ]
    save("02_detection_results.ipynb", cells)


def notebook_03() -> None:
    cells = [
        markdown(
            """
            # 03 — Exact fault-type diagnosis

            This notebook presents the saved multiclass diagnosis results. It does not retrain a model. Binary detection and fault-type diagnosis are separate tasks: first the system asks whether operation is normal or faulty; diagnosis then predicts the original fault label for fault-labelled rows.

            The current diagnosis artefacts are the pooled diagnosis experiment. Building-specific diagnosis is intentionally separate and should only be added when class support is sufficient.
            """
        ),
        code(COMMON),
        markdown("## 1. Load and describe the diagnosis artefacts"),
        code(
            """
RESULTS = ROOT / 'results' / 'diagnosis'
require(RESULTS / 'diagnosis_manifest.json', RESULTS / 'diagnosis_metrics_summary.csv', RESULTS / 'diagnosis_metrics_by_class.csv', RESULTS / 'diagnosis_confusion_matrix.csv')
manifest = read_json(RESULTS / 'diagnosis_manifest.json')
summary = pd.read_csv(RESULTS / 'diagnosis_metrics_summary.csv')
by_class = pd.read_csv(RESULTS / 'diagnosis_metrics_by_class.csv')
confusion = pd.read_csv(RESULTS / 'diagnosis_confusion_matrix.csv')
print('Feature variants:', manifest.get('feature_variants'))
print('Minimum fault rows:', manifest.get('minimum_fault_rows'))
print('Confidence threshold:', manifest.get('confidence_threshold'))
display(summary.round(3))
"""
        ),
        markdown("## 2. Overall diagnosis performance"),
        code(
            """
overall = summary.groupby(['design', 'task'], as_index=False).agg(
    evaluated_rows=('evaluated_rows', 'sum'), unsupported_rows=('unsupported_rows', 'sum'),
    coverage=('coverage', 'mean'), macro_f1=('macro_f1', 'mean'), weighted_f1=('weighted_f1', 'mean'),
    confident_accuracy=('confident_accuracy', 'mean')
)
display(overall.round(3))
"""
        ),
        code(
            """
plot_data = overall[overall['task'].eq('fault_type')].sort_values('macro_f1')
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
ax.barh(plot_data['design'], plot_data['macro_f1'], color=COLORS['dark'])
ax.set_xlim(0, 1)
ax.set_xlabel('Mean macro F1')
ax.set_title('Exact fault-type diagnosis by evaluation design')
plt.show()
"""
        ),
        markdown(
            """
            Macro F1 gives each retained fault type equal importance. Weighted F1 reflects the row distribution and can be higher when common fault types are easier. `unsupported_rows` are reported separately when a class appears in evaluation but was absent from the training fold; they are not silently treated as ordinary model errors.
            """
        ),
        markdown("## 3. Fault-type performance"),
        code(
            """
fault_classes = by_class[by_class['task'].eq('fault_type')].copy()
class_summary = fault_classes.groupby('class_label', as_index=False).agg(
    mean_f1=('f1', 'mean'), mean_precision=('precision', 'mean'), mean_recall=('recall', 'mean'), support=('support', 'sum')
).sort_values('mean_f1')
display(class_summary.round(3))
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(class_summary))), constrained_layout=True)
ax.barh(class_summary['class_label'], class_summary['mean_f1'], color=COLORS['accent'])
ax.set_xlim(0, 1)
ax.set_xlabel('Mean F1')
ax.set_title('Fault-type diagnosis by class')
plt.show()
"""
        ),
        markdown("## 4. Confusion matrix",),
        code(
            """
matrix = confusion[confusion['task'].eq('fault_type')].pivot_table(index='true_label', columns='predicted_label', values='count', aggfunc='sum', fill_value=0)
matrix = matrix.reindex(index=class_summary['class_label'], columns=class_summary['class_label'], fill_value=0)
display(matrix)
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
image = ax.imshow(matrix.to_numpy(), cmap='Blues')
ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=70, ha='right')
ax.set_yticks(range(len(matrix.index)), matrix.index)
ax.set_title('Pooled fault-type confusion matrix')
ax.set_xlabel('Predicted fault type')
ax.set_ylabel('True fault type')
fig.colorbar(image, ax=ax, label='Rows')
plt.show()
"""
        ),
        markdown(
            """
            **Confidence caution.** The saved confidence is a model score used for the configured abstention rule; it is not automatically a calibrated probability. A diagnosis shown at the threshold should remain an inspection aid, not an automatic maintenance decision.
            """
        ),
        markdown("## 5. Diagnosis limitations and next evidence"),
        code(
            """
limitations = pd.DataFrame([
    ['Task boundary', 'Diagnosis predicts fault types; it does not replace binary detection.'],
    ['Class support', 'Rare or unsupported classes are reported separately and should not be over-interpreted.'],
    ['Generalisation', 'These are pooled diagnosis outputs; building-specific diagnosis requires a separate class-support check.'],
    ['Confidence', 'The threshold provides abstention, not a calibrated probability.'],
], columns=['issue', 'interpretation'])
display(limitations)
"""
        ),
    ]
    save("03_diagnosis_results.ipynb", cells)


def notebook_04() -> None:
    cells = [
        markdown(
            """
            # 04 — Operational interpretation and evidence cards

            This notebook translates the saved evaluation results into a staff-readable evidence format. It is not a live dashboard and it does not claim that historical test metrics are live building alerts. The client-facing use is a concise model-evidence card: what was tested, how reliable the result was on that evaluation, and what qualification is required before deployment.
            """
        ),
        code(COMMON),
        markdown("## 1. Load the final evidence artefacts"),
        code(
            """
BUILDING = ROOT / 'results' / 'building_specific'
QUALITY = ROOT / 'results' / 'quality_controlled' / 'exclude_ambiguous_keys'
AUDIT = ROOT / 'results' / 'audit'
require(BUILDING / 'building_detection_test_metrics.csv', BUILDING / 'building_detection_selected_models.csv', QUALITY / 'quality_detection_metrics_summary.csv', AUDIT / 'audit_manifest.json')
test = pd.read_csv(BUILDING / 'building_detection_test_metrics.csv')
selected = pd.read_csv(BUILDING / 'building_detection_selected_models.csv')
quality_summary = pd.read_csv(QUALITY / 'quality_detection_metrics_summary.csv')
audit_manifest = read_json(AUDIT / 'audit_manifest.json')
display(test.round(3))
"""
        ),
        markdown("## 2. Evidence cards for facilities staff"),
        code(
            """
def reliability_label(f1):
    if f1 >= 0.90:
        return 'Strong on this evaluation'
    if f1 >= 0.70:
        return 'Promising; calibrate before use'
    return 'Not reliable for deployment'

cards = test.copy()
cards['evaluation'] = cards['building'] + ' / ' + cards['design'].str.replace('building_', '', regex=False)
cards['evidence_status'] = cards['f1'].map(reliability_label)
cards['staff_message'] = np.where(
    cards['f1'] >= 0.90,
    'Candidate detector for this building and evaluation setting; confirm with target-building calibration.',
    np.where(cards['f1'] >= 0.70, 'Use only as a research aid; collect local normal data and recalibrate.', 'Do not use for operational decisions without redesign and recalibration.')
)
display(cards[['evaluation', 'model', 'evidence_status', 'f1', 'balanced_accuracy', 'false_positive_rate', 'coverage', 'staff_message']].round(3))
"""
        ),
        code(
            """
plot_cards = cards.sort_values('f1')
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
bar_colors = [COLORS['fault'] if value < 0.70 else (COLORS['accent'] if value < 0.90 else COLORS['dark']) for value in plot_cards['f1']]
ax.barh(plot_cards['evaluation'], plot_cards['f1'], color=bar_colors)
ax.set_xlim(0, 1)
ax.set_xlabel('Held-out test F1')
ax.set_title('Evidence-card reliability by building and evaluation')
for index, value in enumerate(plot_cards['f1']):
    ax.text(value + 0.01, index, f'{value:.3f}', va='center')
plt.show()
"""
        ),
        markdown(
            """
            **Recommended staff wording:**

            - `Status: Evaluation evidence` — this is a historical test result, not a live alert.
            - `Building: ...` and `Evaluation: ...` — identify the operating context.
            - `Detection method: ...` — identify the selected model.
            - `Evidence: F1, balanced accuracy, false-positive rate and coverage` — show the supporting numbers.
            - `Operational qualification: ...` — state whether local calibration and human review are required.
            """
        ),
        markdown("## 3. Pooled quality-controlled evidence",),
        code(
            """
pooled_xgb = quality_summary[quality_summary['model'].eq('xgboost')].copy()
display(pooled_xgb[['duplicate_policy', 'design', 'mean_f1', 'mean_balanced_accuracy', 'mean_false_positive_rate', 'mean_coverage']].round(3))
"""
        ),
        markdown(
            """
            The pooled result is useful for comparison across buildings, but it should not be presented as proof that a single model is equally reliable everywhere. The building-specific cards show why local validation is important: hospital AHU-held-out performance is materially weaker than auditorium and office.
            """
        ),
        markdown("## 4. What is and is not delivered",),
        code(
            """
deliverables = pd.DataFrame([
    ['Delivered', 'Reproducible Python audit, cleaning, feature, split and modelling pipeline'],
    ['Delivered', 'Saved pooled and building-specific evaluation metrics'],
    ['Delivered', 'Notebook-based visual evidence and interpretation cards'],
    ['Not delivered', 'A live dashboard or live BMS alerting integration'],
    ['Not delivered', 'A calibrated maintenance decision system'],
    ['Required before deployment', 'Target-building calibration, stakeholder review, alert thresholds and monitoring'],
], columns=['status', 'item'])
display(deliverables)
"""
        ),
        markdown(
            """
            This presentation format is deliberately conservative: it makes the evidence easy to read without hiding the generalisation limits. A facilities team could use the same structure for future live alerts, but the historical dataset alone does not justify calling these cards production alerts.
            """
        ),
    ]
    save("04_operational_outputs.ipynb", cells)


if __name__ == "__main__":
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
    print(f"Wrote four notebooks to {NOTEBOOK_DIR}")
