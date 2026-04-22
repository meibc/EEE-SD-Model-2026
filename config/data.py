"""Data-layer constants used by builders/loaders."""

STATE_TO_DIVISION = {
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1,
    "NJ": 2, "NY": 2, "PA": 2,
    "IL": 3, "IN": 3, "MI": 3, "OH": 3, "WI": 3,
    "IA": 4, "KS": 4, "MN": 4, "MO": 4, "NE": 4, "ND": 4, "SD": 4,
    "DE": 5, "DC": 5, "FL": 5, "GA": 5, "MD": 5, "NC": 5, "SC": 5, "VA": 5, "WV": 5,
    "AL": 6, "KY": 6, "MS": 6, "TN": 6,
    "AR": 7, "LA": 7, "OK": 7, "TX": 7,
    "AZ": 8, "CO": 8, "ID": 8, "MT": 8, "NV": 8, "NM": 8, "UT": 8, "WY": 8,
    "AK": 9, "CA": 9, "HI": 9, "OR": 9, "WA": 9,
}

DEFAULT_CDC_NAMES = [
    "PrEP Eligible",
    "Estimated HIV incidence (MSM)",
    "HIV diagnoses",
    "PrEP",
    "Estimated HIV prevalence (MSM)",
    "HIV viral suppression",
]
