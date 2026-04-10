import numpy as np

from data.unit import Unit
from models.sbm.estimation.jacobian import JacobianEstimator
from models.sbm.estimation.shrinkage import ShrinkageCalculator
from models.sbm.estimation.transforms import FeatureTransformer
from data.data_prep import DataPrep
from engine.results import (
    FitResult,
    FitResults,
    PreparedData,
    PredictionResult,
    PredictionResults,
    RunOutput,
)
from models.sbm.prediction.predictor import Predictor
from config.base import BaseConfig
from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig


class SBRunner:
    def __init__(
        self,
        base_config: BaseConfig,
        opt_config: OptimConfig,
        shrink_config: ShrinkageConfig,
    ):
        self.base = base_config
        self.opt = opt_config
        self.shrink = shrink_config

        self.estimator = JacobianEstimator(opt_config)
        self.shrink_calc = ShrinkageCalculator(shrink_config)
        self.transformer = FeatureTransformer()
        self.data_prep = DataPrep(base_config)
        self.predictor = Predictor()

    def run(
        self,
        fit: bool = True,
        predict: bool = False,
        fit_results: FitResults = None,  # Pass in loaded results
    ) -> RunOutput:
        """
        Run SB model.
        
        Args:
            fit: If True, fit J matrices. If False, must provide fit_results.
            predict: If True, run predictions after fitting.
            fit_results: Pre-loaded fit results (required if fit=False).
        """
        data = self.data_prep.prepare_inputs()

        if fit:
            fit_results = self._fit_all(data)
        elif fit_results is None:
            raise ValueError("Must fit=True or provide fit_results")

        pred_results = self._predict_all(fit_results, data) if predict else None

        return RunOutput(fit=fit_results, predictions=pred_results, inputs=data)

    def _fit_all(self, data: PreparedData) -> FitResults:
        results = {}
        us_J, div_Js = None, {}
        u = np.zeros(data.M.shape[0])  # No exogenous input for now

        for unit in data.units:
            Xbar, SigmaX = self.transformer.transform(unit.values, data.SigmaY)
            J_ref, wJ = self.shrink_calc.get_params(unit, us_J, div_Js)
            J_est = self.estimator.fit(
                Xbar=Xbar,
                SigmaX=SigmaX,
                M=data.M,
                sign_mask=data.sign_mask,
                u=u,
                J_ref=J_ref,
                wJ=wJ,
            )

            results[unit.id] = FitResult(
                unit_id=unit.id,
                J=J_est,
                Ybar=unit.values,
                Xbar=Xbar,
            )

            us_J, div_Js = self._update_hierarchy(unit, J_est, us_J, div_Js)

        return FitResults(results=results, ts=data.ts, v_names=data.v_names)
    
    def _predict_all(self, fit_results: FitResults, data: PreparedData) -> PredictionResults:
        results = {}
        u = np.zeros(data.M.shape[0])
        n_forecast = getattr(self.base, "n_forecast", 5)
        T = len(data.ts)

        for unit_id, fit in fit_results.results.items():
            # Rolling one-step (in-sample): uses observed X at t-1
            Ypred_rolling, Xpred_rolling = self.predictor.predict_rolling(fit.J, fit.Xbar, u)
            
            # Trajectory simulation from X0 using final J
            J_final = fit.J[:, :, -1]
            X0 = fit.Xbar[:, 0]
            n_total = T + n_forecast
            Ypred_trajectory, Xpred_trajectory = self.predictor.predict_trajectory(J_final, X0, u, n_total)

            results[unit_id] = PredictionResult(
                unit_id=unit_id,
                Ypred_rolling=Ypred_rolling,
                Xpred_rolling=Xpred_rolling,
                Ypred_trajectory=Ypred_trajectory,
                Xpred_trajectory=Xpred_trajectory,
            )

        return PredictionResults(results=results, ts=data.ts, v_names=data.v_names)

    def _update_hierarchy(self, unit, J_est, us_J, div_Js):
        # J_est is (m, m, T) — store last timestep for shrinkage
        J_final = J_est[:, :, -1]

        if unit.id in ("USA", "US"):
            us_J = J_final
        elif unit.kind == "division":
            div_Js[unit.census_div] = J_final
        
        return us_J, div_Js
