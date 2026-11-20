// Projection and epidemic model, ported from pipeline/projection.py (project, population_prep),
// models/dynamic/mixture.py (feedback_input), models/scenarios/intervention.py and
// models/epi/prediction/predictor.py. Checked against the Python model by scripts/6_dashboard/validate.py.
(function (root) {
  "use strict";
  const LOGIT_EPS = 0.01, PROB_EPS = 1e-6;
  const POST_YEAR = 2021, RISK_BASELINE_FLOOR = 0.05, RISK_RATIO_BOUNDS = [0.5, 3.0];
  const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const logit = (p, eps) => { const q = clip(p, eps, 1 - eps); return Math.log(q / (1 - q)); };
  const invlogit = (x) => 1 / (1 + Math.exp(-x));

  // Scenario definitions: interventions on measures (by name) and options.
  // effect "last_obs": target = min(2024 level x (1 + delta), baseline); "baseline": target = baseline x (1 + delta).
  function interventions(scn) {
    switch (scn.type) {
      case "reduce_ahs": return { ivs: [{ v: "stigma_ahs", effect: "last_obs", delta: -scn.ahsReduction }] };
      case "prevent_worsening": return { ivs: [{ v: "stigma_ahs", effect: "last_obs", delta: 0 }] };
      case "worsening_climate": return { ivs: [{ v: "stigma_gss", effect: "baseline", delta: 0.6 },
                                               { v: "stigma_family", effect: "baseline", delta: 0.25 }] };
      case "prep_stall": return { ivs: [{ v: "prep_used", effect: "last_obs", delta: 0 }] };
      case "resilient_services": return { ivs: [], gapClosed: scn.gapClosed };
      case "weaken_feedback": return { ivs: [], fbReduction: scn.fbReduction };   // feedback b scaled down
      default: return { ivs: [] };
    }
  }

  function matvec(J, x, m) {
    const y = new Array(m).fill(0);
    for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < m; j++) s += J[i * m + j] * x[j]; y[i] = s; }
    return y;
  }

  // One draw, one fit: baseline (scn null) or scenario trajectory. Returns probabilities [measure][dynamic year]
  // with population PrEP in the PrEP row. Each year: x = J x + u, plus the PrEP -> AHS feedback b x (logit of last
  // year's population PrEP - the state's mean logit of observed PrEP) on AHS, then the interventions hold their
  // targets (capped at the baseline path, the same run without interventions).
  function trajectory(data, state, draw, fbKey, scn) {
    const v = data.v_names, m = v.length, years = data.dyn_years, n = years.length;
    const A = v.indexOf("stigma_ahs"), P = v.indexOf("prep_used");
    const fit = draw.fits[fbKey], J = fit.J, u = fit.u, ref = fit.ref, b = fit.b, r = draw.r;
    const C = state.ceiling;                           // P0 = C x logistic(state), the state's willingness ceiling
    const obs = draw.obs.map((row) => row.slice()), k = obs[0].length;
    const prepObs = obs[P].slice();
    const prepRef = prepObs.reduce((s, p) => s + logit(p, LOGIT_EPS), 0) / k;   // mean logit of observed PrEP
    obs[P] = obs[P].map((p, t) => clip(p / (1 - obs[A][t] * (1 - r)) / C, 0, 0.999));
    const x = obs.map((row, i) => row.map((p) => logit(clip(p, LOGIT_EPS, 1 - LOGIT_EPS), LOGIT_EPS) - ref[i]));
    const def = scn ? interventions(scn) : { ivs: [] };
    const start = years.findIndex((y) => y > state.last_amis_year);
    const steps = scn ? scn.steps : 0;
    const end = Math.min(start + steps, n - 1);
    const scale = (t) => (t < start ? 0 : t >= end ? 1 : (t - start) / Math.max(end - start, 1));
    const ratios = years.map((_, t) => (def.gapClosed != null ? r + scale(t) * def.gapClosed * (1 - r) : r));
    // Weakening the feedback (exploratory): b scaled by 1 - reduction on the interventions' ramp.
    const bAt = (t) => (def.fbReduction ? b * (1 - scale(t) * def.fbReduction) : b);
    const col = (X, t) => X.map((row) => row[t]);
    const probs = (c) => c.map((val, i) => invlogit(val + ref[i]));
    const popPrep = (pr, t) => (t < k ? prepObs[t] : C * pr[P] * (1 - pr[A] * (1 - ratios[t])));
    const ivs = def.ivs.map((iv) => ({ ...iv, idx: v.indexOf(iv.v) }));
    const applyIv = (c, t, iv, anchor, base) => {
      const s = scale(t), p = invlogit(c[iv.idx] + ref[iv.idx]);
      const target = iv.effect === "last_obs" ? Math.min(anchor[iv.idx] * Math.max(0, 1 + iv.delta), base[iv.idx])
                                              : base[iv.idx] * Math.max(0, 1 + iv.delta);
      c[iv.idx] = logit(clip((1 - s) * p + s * target, PROB_EPS, 1 - PROB_EPS), PROB_EPS) - ref[iv.idx];
      return c;
    };
    // pipeline/projection.py project(): probabilities on the model's scale (PrEP row P0 / ceiling)
    function project(list, anchor, base) {
      const X = Array.from({ length: m }, (_, i) => { const row = new Array(n).fill(0); for (let t = 0; t < k; t++) row[t] = x[i][t]; return row; });
      for (let t = k; t < n; t++) {
        const prev = col(X, t - 1);
        let c = matvec(J, prev, m).map((val, i) => val + u[i]);
        c[A] += bAt(t) * (logit(popPrep(probs(prev), t - 1), LOGIT_EPS) - prepRef);
        for (const iv of list) c = applyIv(c, t, iv, anchor, col(base, t));
        c.forEach((val, i) => { X[i][t] = val; });
      }
      return X.map((row, i) => row.map((val) => invlogit(val + ref[i])));
    }
    const finish = (pr) => {                           // population PrEP (observed periods keep observed PrEP)
      const prep = pr[P].map((_, t) => popPrep(col(pr, t), t));
      pr[P] = prep;
      return pr;
    };
    // Without interventions this run is the result (the baseline, or resilient services / weakened feedback,
    // which act through r and b above); with them it is the baseline the targets are anchored to and capped at.
    const base = project([], null, null);
    if (ivs.length === 0) return finish(base);
    return finish(project(ivs, col(base, start - 1), base));
  }

  // Linear interpolation with extrapolation from the end segments (scipy interp1d, fill_value="extrapolate").
  function align(src, vals, tgt) {
    return tgt.map((y) => {
      let i = 0;
      if (y <= src[0]) i = 0; else if (y >= src[src.length - 1]) i = src.length - 2;
      else while (!(y >= src[i] && y <= src[i + 1])) i++;
      const w = (y - src[i]) / (src[i + 1] - src[i]);
      return vals[i] + w * (vals[i + 1] - vals[i]);
    });
  }

  // Epidemic model (models/epi/prediction/predictor.py) for one trajectory.
  function epidemic(data, state, draw, traj) {
    const v = data.v_names, years = data.model_years, T = years.length, p = draw.epi;
    const testing = align(data.dyn_years, traj[v.indexOf("hivtest12")], years).map((x) => clip(x, 1e-6, 1 - 1e-6));
    const prep = align(data.dyn_years, traj[v.indexOf("prep_used")], years);
    const risk = align(data.dyn_years, traj[v.indexOf("risk_behavior")], years);
    const risk0 = Math.max(state.risk0, RISK_BASELINE_FLOOR);
    const incidence = years.map((_, t) => p.beta * state.no_vs[t] * clip(1 - prep[t], 1e-6, 1) *
      Math.pow(clip(Math.max(risk[t], 1e-6) / risk0, RISK_RATIO_BOUNDS[0], RISK_RATIO_BOUNDS[1]), p.alpha));
    const delta = years.map((y, t) => 1 - Math.exp(-p.kdx * testing[t] * (1 + (y >= POST_YEAR ? 1 : 0) * (p.post - 1))));
    const und = new Array(T).fill(0), dx = new Array(T).fill(0);
    und[0] = p.U0; dx[0] = Math.max(0, p.U0 * delta[0]);
    for (let t = 1; t < T; t++) { und[t] = Math.max(0, und[t - 1] + incidence[t - 1] - dx[t - 1]); dx[t] = Math.max(0, und[t] * delta[t]); }
    return { incidence, diagnosed: dx, undiagnosed: und, prepCount: prep.map((x, t) => x * state.n_elig[t]) };
  }

  function runDraw(data, stateKey, d, fbKey, scn) {
    const state = data.states[stateKey], draw = state.draws[d];
    const base = trajectory(data, state, draw, fbKey, null), sc = trajectory(data, state, draw, fbKey, scn);
    return { base, scen: sc, epiBase: epidemic(data, state, draw, base), epiScen: epidemic(data, state, draw, sc) };
  }

  const M = { trajectory, epidemic, runDraw, align };
  if (typeof module !== "undefined" && module.exports) module.exports = M; else root.EEEModel = M;
})(typeof window !== "undefined" ? window : globalThis);
