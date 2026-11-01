import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('rv',Path(__file__).parents[1]/'scripts/readout_validation.py')
rv=importlib.util.module_from_spec(spec);spec.loader.exec_module(rv)

class ReadoutTests(unittest.TestCase):
    def test_repeated_prompts_stay_together(self):
        keys=['math|a','math|a','math|b','idea|a']
        f=rv.fold_ids(keys);self.assertEqual(f[0],f[1])
        self.assertTrue(np.array_equal(f,rv.fold_ids(keys)))
    def test_unseen_domain_uses_training_mean(self):
        y=np.array([[0.,1.],[2.,3.],[4.,5.]])
        means,fallback=rv.references(y,np.array(['a','a','b']))
        r=rv.reference_rows(np.array(['a','new']),means,fallback)
        np.testing.assert_allclose(r,[[1,2],[2,3]])
    def test_full_rank_recovers_affine_signal_on_shifted_inputs(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(300,5))+8
        w=rng.normal(size=(5,3)); y=x@w+np.array([3,7,-2])
        model=rv.fit_rrr(x,y,rank=3,lam=1e-8)
        xt=rng.normal(size=(20,5))+11
        np.testing.assert_allclose(rv.predict(model,xt),xt@w+[3,7,-2],atol=1e-7)
    def test_test_data_cannot_affect_normalization(self):
        x=np.arange(30.).reshape(10,3);y=np.arange(20.).reshape(10,2)
        model=rv.fit_rrr(x,y,rank=2,lam=2)
        before={k:v.copy() for k,v in model.items()}
        rv.predict(model,np.full((3,3),1e9))
        for k in model:np.testing.assert_array_equal(before[k],model[k])

if __name__=='__main__':unittest.main()
