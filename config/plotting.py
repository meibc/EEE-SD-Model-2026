"""Static plotting constants."""

BASELINE_COLOR = "#6f7f8f"
INTERVENTION_COLOR = "#1f5d9b"
OBSERVED_COLOR = "#111111"
FORECAST_BG = "#f2f2f2"

SEM_PLOT_VARS = [
    "stigma_ahs",
    "stigma_gss",
    "stigma_family",
    "out_gid",
    "seehcp",
    "risk_behavior",
    "hivtest12",
    "prep_used",
]

EPI_PLOT_VARS = ["diagnosed", "incidence", "prep_on_count"]

CDC_RAW_MAP = {
    "prep_on_count": "PrEP",
    "incidence": "Estimated HIV incidence (MSM)",
    "diagnosed": "HIV diagnoses",
}
