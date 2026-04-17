"""CLI entrypoint for the project pipeline."""

from config.run import RunConfig
from pipeline.pipeline import run_pipeline


def main(options: RunConfig | None = None) -> dict:
    return run_pipeline(options)


if __name__ == "__main__":
    main()
