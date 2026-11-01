"""Generate revision tables and companion CSVs directly from audited outputs."""
from pathlib import Path
import json,shutil
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P/'revision/results';T=P/'tables';S=P/'supplement'
S.mkdir(exist_ok=True)
names=json.loads((P/'model_names.json').read_text())
def table(name,cols,header,rows):
 (T/(name+'.tex')).write_text('\\begin{tabular}{'+cols+'}\n\\toprule '+header+' \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
d=pd.read_csv(R/'audit/coverage.csv');g=d.groupby('model').sum(numeric_only=True);rows=[]
for m in ['reasoner','gemma4_e4b','gemma4_12b','gemma4_31b','qwen35_4b','qwen35_9b','qwen35_27b']:
 r=g.loc[m];rows.append(f"{names[m]['display']} & {int(r.completed):,} & {int(r.usable):,} & {100*r.usable/r.attempted:.1f} & {int(r.usable_incomplete)} & {100*r.valid_segments/r.annotated_segments:.2f}"+r' \\')
table('coverage','lrrrrr',r'Model & Completed & Usable & Included (\%) & Unfinished & Segments (\%)',rows)
source=P.parent/'docs/paper_synthesis_2026-10-31/sources/ada_corrected_atlas/09_signatures/native_outcome_signature_models.csv'
d=pd.read_csv(source);rows=[]
for domain in ['math','code','gpqa','planning','moral','idea','safety','security']:
 a=d[d.task_type.eq(domain)&d.status.eq('estimated')];metric='bits' if a.outcome_kind.iloc[0]=='binary' else '$\\Delta R^2$'
 fmt=lambda c:f"{a[c].median():+.3f} [{a[c].min():+.3f}, {a[c].max():+.3f}]"
 lab='GPQA' if domain=='gpqa' else domain.title()
 rows.append(f"{lab} & {len(a)}/14 & {metric} & {fmt('amount_gain')} & {fmt('timing_gain')}"+r' \\')
table('outcome_spec','lrlrr',r'Condition & Eligible & Units & Amount: median [range] & Timing: median [range]',rows)
files=[source,*[R/'audit'/n for n in ['coverage.csv','prompt_overlaps.csv','grade_change_counts.csv','grade_changes_by_condition.csv','budget_summary.csv','budget_paired.csv']],
 *[R/'behavior'/n for n in ['fidelity.csv','dose_matched.csv','paired_shifts.csv','similarity_sensitivity.csv','matched_demand_sensitivity.csv','matched_demand_operations.csv']]]
for file in files:shutil.copy2(file,S/file.name)
for file in (R/'audit').glob('*idea_pairwise*.csv'):shutil.copy2(file,S/file.name)
shutil.copy2(R/'audit/open_task_coverage.csv',S/'open_task_coverage.csv')
coverage=pd.read_csv(R/'audit/open_task_coverage.csv');rows=[]
for run,arm,label in [('sched1','sus_math_L19','Checking vector'),('sched1','sus_idea_L19','Exploration vector'),('tx_idea','sig_math','Math-profile vector'),('tx_idea','prompt_math','Math-style prompt')]:
 a=coverage[coverage.run.eq(run)&coverage.comparison.eq(arm)].set_index('condition');r=a.loc[arm];b=a.loc['baseline']
 rows.append(f"{label} & {int(r.unique_keywords)} & {r.seeds} & {int(b.answered)}/{int(b.attempted)} & {int(r.answered)}/{int(r.attempted)} & {int(r.paired_keywords)} ({int(r.paired_answer_rows)})"+r' \\')
table('open_coverage','lrrlll',r'Intervention & Allocated keywords & Seeds & Baseline answers & Arm answers & Paired keywords (rows)',rows)
for file in [R/'readout/validation_summary.csv',R/'multijudge/pairwise_agreement.csv']:
 if file.exists():shutil.copy2(file,S/file.name)
shutil.copy2(R/'within_trace_timing/summary.csv',S/'within_trace_timing.csv')
for label in ['kim','gandhi']:
 file=P.parent/f'src/judge/prompts/{label}_per_sentence.txt';shutil.copy2(file,S/file.name)
readout=R/'readout/validation_summary.csv'
if readout.exists():
 d=pd.read_csv(readout).query('domain=="all"');cv=d[d.split.eq('prompt_cv')];held=d[d.split.eq('heldout_domain')]
 result=(f"Across the 16 targets, mean prompt-balanced MSE reduction over the training-domain reference is {100*cv.skill_over_reference.mean():.2f}\\%. "
   f"With an unseen domain, mean reduction over the training-grand-mean reference is {100*held.skill_over_reference.mean():.2f}\\%; these baselines differ. "
   f"The mean pooled position-level $R^2$ across targets is {held.position_r2.mean():.3f} for held-out domains, indicating poor absolute out-of-domain prediction despite improvement on some components. "
   "Figure~\\ref{fig:readout-validation} gives per-target prompt-bootstrap intervals. Hyperparameters were selected historically, so these fixed-setting checks are not nested estimates of a model-selection procedure.\n"
   "\\begin{figure*}[t]\n\\centering\\includegraphics[width=.95\\textwidth]{figures/figA7_readout_validation.pdf}\n"
   "\\caption{Readout prediction under prompt-disjoint and held-domain validation. Skill compares MSE with the training-domain mean for known domains and the training grand mean for an unseen domain. Whiskers resample prompts; targets are future or next-four-segment rates. The historical steering map remains frozen.}\n"
   "\\label{fig:readout-validation}\n\\end{figure*}\n")
 (T/'readout_result.tex').write_text(result)
print('Built coverage and outcome tables; copied companion audit tables and exact prompts.')
