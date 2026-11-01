from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rl_repair import task_indices

def test_epoch_boundary_uses_next_epoch_permutation():
    import numpy as np
    seed=11;n=5
    expected=np.concatenate([np.random.default_rng(seed+i).permutation(n) for i in range(3)])
    actual=sum([task_indices(s,n,3,seed) for s in range(5)],[])
    assert actual==expected.tolist()

def test_temperature_is_independent_of_sequence_length():
    import pytest
    torch=pytest.importorskip('torch')
    from rl_repair import sampling_logprob
    logits=torch.tensor([[1.,2.,-1.]],requires_grad=True);target=torch.tensor([1])
    short=sampling_logprob(logits,target,.6)
    long=sampling_logprob(logits.repeat(7,1),target.repeat(7),.6)/7
    assert torch.allclose(short,long)
    gs,=torch.autograd.grad(short,logits,retain_graph=True)
    gl,=torch.autograd.grad(long,logits)
    assert torch.allclose(gs,gl)
