import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_existing import paired_budget

def test_bootstrap_independent_of_input_order():
    d=pd.DataFrame([dict(task_type='math',condition=c,budget=100,instance_id=str(i),seed=s,
          correct=int((i+s)%3==0) if c=='baseline' else int(i%2==0),answered=1,tokens=90)
          for c in ['baseline','a','b'] for i in range(10) for s in [1,2]])
    pd.testing.assert_frame_equal(paired_budget(d,reps=50),paired_budget(d.sample(frac=1,random_state=17),reps=50))
