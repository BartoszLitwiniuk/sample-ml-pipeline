import pandas as pd

from data.etl import DataPreparation


def _make_raw_dataset():
    """Minimal DataFrame with columns expected by DataPreparation."""
    return pd.DataFrame(
        {
            "Numtppd": [0, 1, 2, 0],
            "Numtpbi": [0, 1, 0, 1],
            "Indtppd": [0, 1, 0, 1],
            "Indtpbi": [0, 0, 1, 1],
            "CalYear": [2014, 2014, 2015, 2015],
            "Gender": ["M", "F", "M", "F"],
            "Type": ["A", "A", "B", "B"],
            "Category": ["C1", "C1", "C2", "C2"],
            "Occupation": ["O1", "O2", "O1", "O2"],
            "SubGroup2": ["S1", "S1", "S2", "S2"],
            "Group2": ["G1", "G2", "G1", "G2"],
            "Group1": ["H1", "H1", "H2", "H2"],
        }
    )


class TestDataPreparationTransformDataset:
    """Tests for DataPreparation.transform_dataset."""

    def test_transform_dataset(self):
        # given
        df = _make_raw_dataset()

        # when
        result = DataPreparation.transform_dataset(df.copy())

        # then
        for col in DataPreparation.NUMERICAL_COLUMNS:
            assert col not in result.columns

        assert DataPreparation.TARGET_COLUMN in result.columns
        assert result[DataPreparation.TARGET_COLUMN].tolist() == [0, 1, 1, 0]
        assert DataPreparation.TARGET_COLUMN in result.columns


class TestDataPreparationTrainTestSplit:
    """Tests for DataPreparation.train_test_split."""

    def test_train_test_split(self):
        """train_test_split returns X_train, X_test, y_train, y_test."""
        # given
        df = _make_raw_dataset()
        transformed = DataPreparation.transform_dataset(df.copy())

        # when
        out = DataPreparation.train_test_split(
            transformed, test_size=0.25, random_state=42
        )

        # then
        X_train, X_test, y_train, y_test = out
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        assert len(X_train) == 3
        assert len(y_train) == 3
        assert len(X_test) == 1
        assert len(y_test) == 1
