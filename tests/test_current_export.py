"""Scientific handoff checks: geography alignment and incompatible-fit rejection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('current_export',Path(__file__).resolve().parents[1]/'scripts/export_current_research.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=dict(schema_version=np.array([9]),geo_ids=np.array(['GA','TN','CA']),
            model_years=np.arange(2017,2037),cdc_native_years=np.arange(2017,2023),
            sem_obs_years=np.array([2018,2020,2022,2024]),sem_pred_years=np.array([2018,2020,2022,2024]),
            sem_v_names=np.array(['a','b']),cdc_raw_names=np.array(['x']),
            sem_obs=np.full((3,2,4),0.3),sem_pred=np.full((3,2,4),0.3),
            sem_fit_J_last=np.zeros((3,2,2)),sem_fit_drift=np.zeros((3,2)),
            sem_J_samples=np.zeros((2,3,2,2)),sem_drift_samples=np.zeros((2,3,2)),
            sem_reference_probs=np.array([0.1,0.1]),cdc_raw=np.zeros((3,1,20)),
            cdc_raw_native=np.zeros((3,1,6)),cdc_kappa_prep=np.array([10.,20.,30.]),
            cdc_risk0=np.ones(3))
        for key in ['cdc_beta','cdc_alpha','cdc_kdx','cdc_U0','cdc_post_multiplier']:
            self.base[key]=np.array([[1.,2.,3.],[4.,5.,6.]])
        self.fit=dict(selected_states=np.array(['CA','GA']),unit_order=np.array(['GA','CA']),
            v_names=np.array(['a','b']),ts=np.array([2018,2020,2022,2024]),
            Jmeans_stack=np.array([[np.eye(2)*.7,np.eye(2)*.8]]*2),
            drift_stack=np.zeros((2,2,2)),success_stack=np.ones((2,2),dtype=bool),
            reference_probs=np.array([.2,.25]))
    def tearDown(self):self.temp.cleanup()
    def run_export(self):
        a=self.root/'base.npz';b=self.root/'fit.npz';o=self.root/'current.npz'
        np.savez(a,**self.base);np.savez(b,**self.fit)
        module.export(a,b,o);return np.load(o)
    def test_subset_and_order_preserve_matching_epidemiological_draws(self):
        with self.run_export() as out:
            self.assertEqual(out['geo_ids'].tolist(),['CA','GA'])
            np.testing.assert_array_equal(out['cdc_beta'],self.base['cdc_beta'][:,[2,0]])
            np.testing.assert_array_equal(out['sem_J_samples'],self.fit['Jmeans_stack'][:,[1,0]])
            np.testing.assert_array_equal(out['sem_reference_probs'],self.fit['reference_probs'])
            np.testing.assert_allclose(out['sem_pred'][:,:,0],out['sem_obs'][:,:,0])
    def test_reject_variable_order_mismatch(self):
        self.fit['v_names']=np.array(['b','a'])
        with self.assertRaisesRegex(ValueError,'variable order'):self.run_export()
    def test_reject_failed_fits(self):
        self.fit['success_stack'][0,0]=False
        with self.assertRaisesRegex(ValueError,'optimizer failures'):self.run_export()

if __name__=='__main__':unittest.main()
