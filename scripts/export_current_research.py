#!/usr/bin/env python3
"""Build the accepted research bundle from completed empirical-reference fits.

Only refitted primary states are exported: no mixing of old coefficients with
new reference probabilities. Existing epidemiological posterior draws are kept.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]

def export(base_path: Path, fit_path: Path, output_path: Path) -> dict:
    with np.load(base_path, allow_pickle=True) as d:
        arrays = {k: d[k] for k in d.files}
    with np.load(fit_path, allow_pickle=True) as d:
        fit = {k: d[k] for k in d.files}
    states = fit['selected_states'].astype(str).tolist()
    geos = arrays['geo_ids'].astype(str).tolist()
    units = fit['unit_order'].astype(str).tolist()
    if len(states) != len(set(states)):
        raise ValueError('Duplicate selected states')
    if arrays['sem_v_names'].astype(str).tolist() != fit['v_names'].astype(str).tolist():
        raise ValueError('SEM variable order mismatch')
    if not np.array_equal(arrays['sem_obs_years'], fit['ts']):
        raise ValueError('SEM observation year mismatch')
    ix = [geos.index(s) for s in states]
    fx = [units.index(s) for s in states]
    if not np.all(fit['success_stack'][:, fx]):
        raise ValueError('Primary state fits contain optimizer failures')
    geo_axis0 = {'geo_ids','sem_obs','sem_pred','sem_fit_J_last','sem_fit_drift',
                 'cdc_raw','cdc_raw_native','cdc_kappa_prep','cdc_risk0'}
    geo_axis1 = {'sem_J_samples','sem_drift_samples','cdc_beta','cdc_alpha',
                 'cdc_kdx','cdc_U0','cdc_post_multiplier'}
    global_keys = {'schema_version','model_years','cdc_native_years','sem_obs_years',
                   'sem_pred_years','sem_v_names','cdc_raw_names','sem_reference_probs'}
    unknown = set(arrays) - geo_axis0 - geo_axis1 - global_keys
    if unknown:
        raise ValueError(f'Unrecognized bundle fields: {sorted(unknown)}')
    for key in geo_axis0 & arrays.keys():
        arrays[key] = arrays[key][ix]
    for key in geo_axis1 & arrays.keys():
        arrays[key] = arrays[key][:, ix]
    arrays['sem_J_samples'] = fit['Jmeans_stack'][:, fx]
    arrays['sem_drift_samples'] = fit['drift_stack'][:, fx]
    arrays['sem_fit_J_last'] = np.median(arrays['sem_J_samples'], axis=0)
    arrays['sem_fit_drift'] = np.median(arrays['sem_drift_samples'], axis=0)
    arrays['sem_reference_probs'] = fit['reference_probs']
    refs = arrays['sem_reference_probs']
    if not np.all((refs > 0) & (refs < 1)):
        raise ValueError('Reference probabilities must lie inside (0,1)')
    # Rebuild cached trajectories under the same fitted parameters and references.
    for g in range(len(states)):
        x = logit(np.clip(arrays['sem_obs'][g, :, 0], 1e-6, 1-1e-6))-logit(refs)
        for t in range(len(arrays['sem_pred_years'])):
            arrays['sem_pred'][g, :, t] = expit(x + logit(refs))
            x = arrays['sem_fit_J_last'][g] @ x + arrays['sem_fit_drift'][g]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    manifest = {
        'specification': 'empirical_p05_joint_respondent_bootstrap',
        'states': states, 'sem_draws': int(arrays['sem_J_samples'].shape[0]),
        'base_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
        'fit_sha256': hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
        'note': 'Research primary states only. Not an all-state dashboard release.'
    }
    output_path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'output/standardized_input_v9.npz')
    p.add_argument('--fit',type=Path,default=ROOT/'output/empirical_p05_reference_sensitivity_n500/fits/empirical_p05_references.npz')
    p.add_argument('--output',type=Path,default=ROOT/'output/current/standardized_research.npz')
    a=p.parse_args(); print(json.dumps(export(a.base,a.fit,a.output),indent=2))
if __name__=='__main__':main()
