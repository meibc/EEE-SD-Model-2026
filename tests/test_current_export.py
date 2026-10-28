"""Scientific handoff checks: geography alignment and incompatible-fit rejection."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artifacts.research_bundle import build_research_bundle
from data.state_inputs import StateInputs

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
            cdc_raw=np.zeros((3,1,20)),
            cdc_raw_native=np.zeros((3,1,6)),cdc_kappa_prep=np.array([10.,20.,30.]),
            cdc_risk0=np.ones(3))
        for key in ['cdc_beta','cdc_alpha','cdc_kdx','cdc_U0','cdc_post_multiplier']:
            self.base[key]=np.array([[1.,2.,3.],[4.,5.,6.]])
        self.fit=dict(selected_states=np.array(['CA','GA']),unit_ids=np.array(['GA','CA']),
            v_names=np.array(['a','b']),years=np.array([2018,2020,2022,2024]),
            J=np.array([[np.eye(2)*.7,np.eye(2)*.8]]*2),
            drift=np.zeros((2,2,2)),success_stack=np.ones((2,2),dtype=bool),
            anchor_logits=np.arange(8.).reshape(2,2,2),input_draw_indices=np.array([1,0]))
    def tearDown(self):self.temp.cleanup()
    def run_export(self):
        a=self.root/'base.npz';b=self.root/'fit.npz';o=self.root/'current.npz'
        d=self.root/'draws.npz'
        np.savez(a,**self.base);np.savez(b,**self.fit)
        self.draws=np.random.default_rng(1).uniform(.1,.9,(3,2,2,4))  # draw x unit x variable x period
        np.savez(d,observed_draws=self.draws,observed_point=np.full((2,2,4),0.3),unit_ids=np.array(['GA','CA']),
            unit_kinds=np.array(['state','state']),census_divisions=np.array([5,9]),sample_sizes=np.array([100.,200.]),
            covariance=np.eye(2),v_names=np.array(['a','b']),years=np.array([2018,2020,2022,2024]),source_draw_ids=np.arange(3))
        build_research_bundle(a,b,o,d);return np.load(o)
    def test_subset_and_order_preserve_matching_epidemiological_draws(self):
        with self.run_export() as out:
            self.assertEqual(out['geo_ids'].tolist(),['CA','GA'])
            np.testing.assert_array_equal(out['cdc_beta'],self.base['cdc_beta'][:,[2,0]])
            np.testing.assert_array_equal(out['sem_J_samples'],self.fit['J'][:,[1,0]])
            np.testing.assert_array_equal(out['sem_anchor_logits'],self.fit['anchor_logits'][:,[1,0]])
            np.testing.assert_allclose(out['sem_pred'][:,:,0],out['sem_obs'][:,:,0])
            # Observed draws follow the fit's source draws, in selected-state order (CA, GA).
            np.testing.assert_array_equal(out['sem_obs_samples'][0],self.draws[1][[1,0]])
    def test_reject_variable_order_mismatch(self):
        self.fit['v_names']=np.array(['b','a'])
        with self.assertRaisesRegex(ValueError,'do not match the fit'):self.run_export()
    def test_input_file_round_trip(self):
        self.run_export();inputs=StateInputs.load(self.root/'draws.npz')
        self.assertEqual([(u.id,u.kind,u.census_div,u.sample_size) for u in inputs.units],[('GA','state',5,100.),('CA','state',9,200.)])
    def test_epidemiological_base_without_sem_fields(self):
        for key in [k for k in self.base if k.startswith('sem_')]:del self.base[key]
        with self.run_export() as out:
            self.assertEqual(out['sem_v_names'].tolist(),['a','b'])
            np.testing.assert_array_equal(out['sem_obs_years'],self.fit['years'])
            np.testing.assert_allclose(out['sem_pred'][:,:,0],out['sem_obs'][:,:,0])
    def test_reject_failed_fits(self):
        self.fit['success_stack'][0,0]=False
        with self.assertRaisesRegex(ValueError,'optimizer failures'):self.run_export()

if __name__=='__main__':unittest.main()
