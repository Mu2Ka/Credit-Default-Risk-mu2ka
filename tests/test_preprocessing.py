import unittest

import pandas as pd

from preprocessing import fill_missing, get_train_test_data


def example_data():
    return pd.DataFrame({
        "DAYS_LAST_PHONE_CHANGE": [-10, -20],
        "CNT_FAM_MEMBERS": [2, 2],
        "AMT_ANNUITY": [10, 20],
        "AMT_GOODS_PRICE": [100, 200],
        "EXT_SOURCE_1": [0.2, 0.4],
        "EXT_SOURCE_2": [0.3, 0.5],
        "EXT_SOURCE_3": [0.4, 0.6],
        "DAYS_EMPLOYED": [-100, -200],
        "AMT_CREDIT": [100, 200],
        "AMT_INCOME_TOTAL": [50, 100],
        "DAYS_BIRTH": [-3650, -7300],
        "CATEGORY": ["A", "B"],
    })


class PreprocessingTest(unittest.TestCase):
    def test_missing_value_uses_training_mean(self):
        train = example_data()
        test = example_data().iloc[:1].copy()
        test["EXT_SOURCE_1"] = None

        _, prepared_test = fill_missing(train, test)

        self.assertAlmostEqual(prepared_test.loc[0, "EXT_SOURCE_1"], 0.3)
        self.assertAlmostEqual(prepared_test.loc[0, "credit_to_annuity"], 10)
        self.assertTrue(pd.isna(test.loc[0, "EXT_SOURCE_1"]))

    def test_split_keeps_features_and_targets_aligned(self):
        data = pd.concat([example_data()] * 5, ignore_index=True)
        data["TARGET"] = [0, 1] * 5

        x_train, x_test, y_train, y_test, *_ = get_train_test_data(data)

        self.assertEqual(len(x_train), 8)
        self.assertEqual(len(x_test), 2)
        self.assertTrue(x_train.index.equals(y_train.index))
        self.assertTrue(x_test.index.equals(y_test.index))
        self.assertFalse(x_train.isna().any().any())
        self.assertFalse(x_test.isna().any().any())
        self.assertEqual(list(x_train.columns), list(x_test.columns))


if __name__ == "__main__":
    unittest.main()
