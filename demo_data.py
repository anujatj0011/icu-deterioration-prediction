"""Public portfolio demo data.

IMPORTANT: These profiles are hand-authored synthetic examples. They are not
sampled, transformed, aggregated, or otherwise derived from MIMIC-IV records.
The public demo intentionally does not load the MIMIC-trained model.
"""
from __future__ import annotations

import pandas as pd


PROFILES = [
    {
        "display_id": "SYN-001", "calibrated_probability": 0.78,
        "age": 76, "heart_rate_last": 118, "sbp_last": 82, "resp_rate_last": 29,
        "spo2_last": 91, "lactate_last": 5.2, "creatinine_last": 2.1,
        "platelets_last": 112, "gcs_verbal_last": 3,
        "drivers_up": ["Elevated lactate", "Low systolic blood pressure", "High respiratory rate"],
        "drivers_down": ["Oxygen saturation not severely reduced"],
    },
    {
        "display_id": "SYN-002", "calibrated_probability": 0.61,
        "age": 68, "heart_rate_last": 104, "sbp_last": 94, "resp_rate_last": 25,
        "spo2_last": 93, "lactate_last": 3.6, "creatinine_last": 1.7,
        "platelets_last": 145, "gcs_verbal_last": 4,
        "drivers_up": ["Elevated lactate", "Lower systolic blood pressure", "Tachycardia"],
        "drivers_down": ["Preserved verbal GCS"],
    },
    {
        "display_id": "SYN-003", "calibrated_probability": 0.47,
        "age": 59, "heart_rate_last": 96, "sbp_last": 103, "resp_rate_last": 23,
        "spo2_last": 95, "lactate_last": 2.8, "creatinine_last": 1.3,
        "platelets_last": 171, "gcs_verbal_last": 4,
        "drivers_up": ["Mildly elevated lactate", "Respiratory rate"],
        "drivers_down": ["Systolic blood pressure", "Platelet count"],
    },
    {
        "display_id": "SYN-004", "calibrated_probability": 0.34,
        "age": 51, "heart_rate_last": 88, "sbp_last": 112, "resp_rate_last": 20,
        "spo2_last": 96, "lactate_last": 2.1, "creatinine_last": 1.1,
        "platelets_last": 196, "gcs_verbal_last": 5,
        "drivers_up": ["Mild lactate elevation"],
        "drivers_down": ["Preserved blood pressure", "Normal verbal GCS"],
    },
    {
        "display_id": "SYN-005", "calibrated_probability": 0.22,
        "age": 43, "heart_rate_last": 82, "sbp_last": 124, "resp_rate_last": 18,
        "spo2_last": 97, "lactate_last": 1.6, "creatinine_last": 0.9,
        "platelets_last": 224, "gcs_verbal_last": 5,
        "drivers_up": ["Age contribution"],
        "drivers_down": ["Stable blood pressure", "Lower lactate", "Normal verbal GCS"],
    },
    {
        "display_id": "SYN-006", "calibrated_probability": 0.12,
        "age": 35, "heart_rate_last": 76, "sbp_last": 132, "resp_rate_last": 16,
        "spo2_last": 98, "lactate_last": 1.2, "creatinine_last": 0.8,
        "platelets_last": 251, "gcs_verbal_last": 5,
        "drivers_up": [],
        "drivers_down": ["Stable vital signs", "Lower lactate", "Normal verbal GCS"],
    },
]


def demo_frame() -> pd.DataFrame:
    return pd.DataFrame(PROFILES).sort_values(
        "calibrated_probability", ascending=False, ignore_index=True
    )
