import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from confirmation import boxed,canonical,ARMS
from analyze_confirmation import holm,paired_effects

def test_references_and_overlap_normalization():
    assert boxed(r'First \boxed{2}, finally \boxed{\frac{1}{3}}.')==r'\frac{1}{3}'
    assert boxed(r'\boxed{unfinished') is None
    assert canonical('  x + ２ \n')==canonical('X+2')
    assert canonical('x+2')!=canonical('x+3')

def test_holm_and_seed_pairing():
    assert np.allclose(holm([.04,.01]),[.04,.02])
    rows=[]
    for arm in ARMS:
        for i in range(8):
            for sample in [0,1]:
                v=int(arm!='baseline' and sample==0)
                rows.append(dict(condition=arm,instance_id=str(i),sample=sample,natural_correct=v,elicited_correct=v,answered=1,completed=1,hard_loop=0,tokens=100))
    d=pd.DataFrame(rows);result=paired_effects(d,reps=100)
    assert np.allclose(result.query('metric=="natural_correct"').delta,.5)
    assert result.query('metric=="natural_correct"').n_tasks.eq(8).all()
    assert result.query('metric!="natural_correct"').p_holm_primary.isna().all()
    with pytest.raises(AssertionError):paired_effects(d.iloc[1:],reps=100)
