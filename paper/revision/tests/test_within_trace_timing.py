from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from within_trace_timing import phase_draw,PATTERNS

def test_joint_permutation_preserves_counts_and_phase_sizes():
    counts=np.zeros(256,dtype=int);counts[[0,1,3,128,255]]=[4,3,5,2,1]
    draw=phase_draw(counts,[6,4,5],np.random.default_rng(4))
    assert np.array_equal(draw.sum(0),counts)
    assert np.array_equal(draw.sum(1),[6,4,5])
    assert np.array_equal((draw@PATTERNS).sum(0),counts@PATTERNS)

def test_empty_phase_retains_other_positions():
    counts=np.zeros(256,dtype=int);counts[3]=1
    draw=phase_draw(counts,[1,0,0],np.random.default_rng(2))
    assert draw[0,3]==1 and draw[1:].sum()==0
