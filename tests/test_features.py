import unittest

import pandas as pd

from features import build_bureau_features


class BureauFeaturesTest(unittest.TestCase):
    def test_debt_ratio_is_calculated_per_client(self):
        bureau = pd.DataFrame({
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [11, 12, 13],
            "DAYS_CREDIT_ENDDATE": [10, -1, 2],
            "DAYS_ENDDATE_FACT": [5, 0, 1],
            "AMT_CREDIT_SUM_DEBT": [20, 10, 0],
            "AMT_CREDIT_SUM": [100, 100, 0],
            "DAYS_CREDIT": [-10, -20, -30],
            "CREDIT_DAY_OVERDUE": [0, 2, 0],
            "CREDIT_TYPE": ["Consumer credit", "Credit card", "Consumer credit"],
        })

        result = build_bureau_features(bureau).set_index("SK_ID_CURR")

        self.assertEqual(result.loc[1, "CREDIT_COUNT"], 2)
        self.assertAlmostEqual(result.loc[1, "CREDIT_DEBT_RATIO"], 0.15)
        self.assertTrue(pd.isna(result.loc[2, "CREDIT_DEBT_RATIO"]))


if __name__ == "__main__":
    unittest.main()
