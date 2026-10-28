
"""Transparent preparation quantity recommendations."""
import math

def make_recommendation(predicted_attendance: float, safety_margin_pct: float = 5.0,
                        planned_quantity: int = 0) -> dict:
    if not 0 <= safety_margin_pct <= 30:
        raise ValueError("Safety margin must be between 0% and 30%.")
    if planned_quantity < 0:
        raise ValueError("Planned quantity cannot be negative.")
    predicted = max(0, int(round(float(predicted_attendance))))
    prepare = int(math.ceil(predicted * (100 + safety_margin_pct) / 100))
    difference = int(planned_quantity) - predicted
    return {
        "predicted_attendance": predicted,
        "recommended_prepare": prepare,
        "planned_minus_predicted": difference,
        "estimated_surplus": max(0, difference),
        "estimated_shortage": max(0, -difference),
        "note": "Surplus/shortage is an estimate relative to the entered plan, not measured food waste."
    }
