import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.scenario import route_options, voyage_track
from prediction.predictor import calculate_compliance_forecast, predict_fuel_consumption


class GreenFleetSmokeTests(unittest.TestCase):
    def test_prediction_and_compliance(self):
        fuel = predict_fuel_consumption("Container Ship", 10000, 35000, 3600, 18, fuel_type="LNG")
        self.assertGreater(fuel, 0)
        compliance = calculate_compliance_forecast(1000, 10000, 3600, 18)
        self.assertIn(compliance["status"], {"green", "amber", "red"})

    def test_route_options_and_tracking(self):
        routes = route_options(route_id="R03", fuel_type="LNG")
        self.assertEqual(len(routes["options"]), 3)
        self.assertIn(routes["recommended_route_id"], {item["id"] for item in routes["options"]})
        tracking = voyage_track(route_id="R03", progress_pct=25, fuel_type="LNG")
        self.assertEqual(tracking["status"], "tracking")
        self.assertGreater(tracking["fuel_remaining_tonnes"], 0)


if __name__ == "__main__":
    unittest.main()