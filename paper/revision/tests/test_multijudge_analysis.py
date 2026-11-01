import importlib.util
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_multijudge import agreement

def test_rare_label_disagreement_not_hidden_by_negatives():
    a=agreement([980,10,10,0])
    assert a['agreement']==.98
    assert a['positive_agreement']==0
    assert a['kappa']<0

def test_absent_label_does_not_claim_perfect_positive_agreement():
    a=agreement([100,0,0,0])
    assert np.isnan(a['positive_agreement'])
    assert np.isnan(a['kappa'])

def test_symmetric_agreement():
    assert agreement([20,3,5,7])==agreement([20,5,3,7])
