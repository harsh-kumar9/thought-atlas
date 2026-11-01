"""Ada: reconcile stored cutoff outcomes with current extraction grades.

No generation or new grading. Cutoff-truncated outcomes retain their existing
direct grade, exactly as mc.budget.outcomes; completed traces use current
mc.budget.load_grades(). Outputs stay outside canonical experiment directories.
"""
import hashlib,json,sys
from pathlib import Path
import polars as pl
from mode_control.mc.common import DATA
from mode_control.mc.budget import load_grades, paired

out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
grades=load_grades();parts=[];audit=[];inputs=[]
for run,arms in [('p1_budget',['baseline','ol_math_A2']),('p4_eval3',['rl_zeroT_s60']),('p4_eval2',['rl_a2initT_s60'])]:
    rd=DATA/'runs'/run
    tr=[]
    for arm in arms:
        p=rd/f'traces_{arm}.parquet';inputs.append(p)
        tr.append(pl.read_parquet(p,columns=['trace_id','condition','instance_id','seed','n_new_tokens','run_fingerprint']))
    tr=pl.concat(tr)
    if 'fingerprints' not in locals():fingerprints=set(tr['run_fingerprint'])
    assert set(tr['run_fingerprint'])==fingerprints
    p=rd/'analysis/budget_outcomes.parquet';inputs.append(p)
    old=pl.read_parquet(p).filter(pl.col('condition').is_in(arms))
    d=old.join(tr,on=['condition','instance_id','seed'],validate='m:1')
    assert d.height==old.height
    current=[];changes=0;available=0
    for row in d.iter_rows(named=True):
        g=grades.get(row['trace_id'])
        use=row['n_new_tokens']<=row['budget'] and g is not None
        value=int(g) if use else row['correct']
        available+=use;changes+=value!=row['correct'];current.append(value)
    d=d.with_columns(pl.Series('correct',current)).select(old.columns)
    audit.append(dict(run=run,rows=d.height,completed_rows_with_current_grade=available,changed_cutoff_outcomes=changes))
    parts.append(d)
d=pl.concat(parts);d.write_parquet(out/'rl_budget_outcomes.parquet')
summary=d.group_by('condition','task_type','budget').agg(pl.len().alias('n'),pl.col('correct').mean(),pl.col('answered').mean(),pl.col('tokens').mean()).sort('condition','budget')
summary.write_csv(out/'rl_budget_summary.csv');paired(d,'baseline').write_csv(out/'rl_budget_paired.csv')
inputs+=sorted(DATA.glob('extract_free*/grades.parquet'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(audit=audit,grade_policy='current extraction grade for traces ending by cutoff; stored direct grade for truncated prefixes',
              source_sha256=digest(Path(__file__)),inputs=[dict(path=str(p),sha256=digest(p)) for p in inputs])
(out/'rl_grade_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(audit,indent=2));print(summary.filter(pl.col('budget').is_in([8192,32768])))
