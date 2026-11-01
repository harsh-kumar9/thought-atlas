"""Task-level uncertainty must not let decoding-seed counts reweight tasks."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_elicitation import paired

def test_seed_averaging_precedes_task_difference():
    rows=[]
    for condition,values in [('baseline',[(0,0),(0,)]),('arm',[(1,1),(0,)])]:
        for item,seeds in enumerate(values):
            for seed,value in enumerate(seeds):
                rows.append(dict(task_type='math',budget=100,condition=condition,instance_id=str(item),seed=seed,
                    correct=value,answered=1,finished=True,total_generated_and_cue_tokens=90))
    result=paired(pd.DataFrame(rows),reps=100)
    r=result[result.metric.eq('correct')].iloc[0]
    assert r.n_tasks==2 and r.delta==.5

def test_missing_intervention_task_is_not_silently_dropped():
    d=pd.DataFrame([dict(task_type='math',budget=100,condition=c,instance_id=i,correct=1,answered=1,
        finished=True,total_generated_and_cue_tokens=90) for c,i in [('baseline','a'),('baseline','b'),('arm','a')]])
    import pytest
    with pytest.raises(AssertionError):paired(d,reps=10)
