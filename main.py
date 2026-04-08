from pathlib import Path
import pickle

from config.base import BaseConfig
from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig
from models.sbm.runner import SBRunner
from pipeline.results import RunOutput


def main():
    # === Config ===
    base = BaseConfig()
    opt = OptimConfig()
    shrink = ShrinkageConfig()

    # === Options (edit here) ===
    load_fits = False
    save_fits = True
    run_predict = True
    output_dir = Path("output")

    # === Run ===
    runner = SBRunner(base, opt, shrink)

    if load_fits:
        loaded = load(output_dir / "output.pkl")
        if isinstance(loaded, RunOutput):
            fit_results = loaded.fit
        elif isinstance(loaded, dict) and "fit" in loaded:
            fit_results = loaded["fit"]
        else:
            raise ValueError("Loaded object does not contain fit results")
        output = runner.run(fit=False, predict=run_predict, fit_results=fit_results)
    else:
        output = runner.run(fit=True, predict=run_predict)

    # === Save ===
    if save_fits:
        output_dir.mkdir(exist_ok=True)
        save(output, output_dir / "output.pkl")
        print(f"Saved to {output_dir / 'output.pkl'}")


def save(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
