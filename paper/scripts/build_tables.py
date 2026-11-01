"""Build manuscript tables directly from the frozen analysis inputs."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parents[1]
T=P/'tables';T.mkdir(exist_ok=True)
names=json.loads((P/'model_names.json').read_text())
g=pd.read_csv(P/'data/figure_inputs/gains.csv').drop_duplicates('gen_model').set_index('gen_model')
rows=[]
for key in ['reasoner','gemma4_e4b','gemma4_12b','gemma4_31b','qwen35_4b','qwen35_9b','qwen35_27b','anchor','r1qwen7b','openthinker3_7b']:
 m=names[key]; n=f"{int(g.loc[key,'n']):,}" if key in g.index else '--'
 rows.append(m['display']+r' & \path{'+m['checkpoint']+'} & '+n+r' \\')
(T/'models.tex').write_text('\n'.join(rows)+'\n')
x=pd.read_csv(P/'revision/results/audit/budget_paired.csv')
rows=[]
for budget in [12288,32768]:
 for source,target in [('math','gpqa'),('math','planning'),('moral','math'),('idea','math'),('gpqa','math'),('gpqa','gpqa'),('planning','planning')]:
  r=x[(x.budget==budget)&(x.condition=='sig_'+source)&(x.task_type==target)].iloc[0]
  lab=lambda s: 'GPQA' if s=='gpqa' else s.title()
  rows.append(f"{lab(source)} $\\to$ {lab(target)} & {budget:,} & {int(r.n_tasks)} & {100*r.d_correct:+.1f} [{100*r.d_correct_lo:+.1f}, {100*r.d_correct_hi:+.1f}] & {100*r.d_answered:+.1f} & {r.d_tokens:+,.0f}"+r' \\')
(T/'transfer.tex').write_text('\n'.join(rows)+'\n')
macros=[]
for name,target in [('MathGPQAShort','gpqa'),('MathPlanningShort','planning')]:
 r=x[x.budget.eq(12288)&x.condition.eq('sig_math')&x.task_type.eq(target)].iloc[0]
 value=f'{100*r.d_correct:.1f} pp [{100*r.d_correct_lo:.1f}, {100*r.d_correct_hi:.1f}]'
 macros.append('\\newcommand{\\'+name+'}{'+value+'}')
(T/'result_macros.tex').write_text('\n'.join(macros)+'\n')
rows=[]
specs=[('sched1','sus_math_L19','Checking vector'),('sched1','sus_idea_L19','Exploration vector'),('tx_idea','sig_math','Math-profile vector'),('tx_idea','prompt_math','Math-style prompt')]
for file,judge in [('pairwise_gemma','Gemma4-31B-it'),('pairwise_qwen','Qwen3.5-27B')]:
 d=pd.read_csv(P/f'data/figure_inputs/{file}.csv')
 for run,arm,label in specs:
  a=d[(d.run==run)&(d.condition==arm)].set_index('axis'); o=a.loc['originality']; f=a.loc['feasibility']
  fmt=lambda r:f"{r.net_win:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
  rows.append(f"{label} & {judge} & {int(o.n_keywords)} & {fmt(o)} & {fmt(f)}"+r' \\')
(T/'open_quality.tex').write_text('\n'.join(rows)+'\n')
print('Built model, paired transfer, and open-quality tables.')

layouts={
 'models':('llr',r'Model name in the paper & Exact checkpoint identifier & Atlas traces'),
 'transfer':('lrrlrr',r'Source $\to$ target & Cutoff & Tasks & Accuracy $\Delta$ [95\% CI] & Answer $\Delta$ & Token $\Delta$'),
 'open_quality':('llrll',r'Intervention & Judge & Keywords & Originality [95\% CI] & Feasibility [95\% CI]')}
for name,(columns,heading) in layouts.items():
 file=T/(name+'.tex');body=file.read_text()
 file.write_text('\\begin{tabular}{'+columns+'}\n\\toprule '+heading+' \\\\\n\\midrule\n'+body+'\\bottomrule\n\\end{tabular}\n')
