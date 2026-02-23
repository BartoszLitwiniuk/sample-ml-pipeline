import pandas as pd
from sklearn.model_selection import train_test_split
from utils.custom_logger import logger
from utils.seed import DEFAULT_SEED

class DataPreparation:

    TARGET_COLUMN = 'target'
    CATEGORICAL_COLUMNS = ['CalYear', 'Gender', 'Type', 'Category', 'Occupation', 'SubGroup2', 'Group2', 'Group1']
    NUMERICAL_COLUMNS = ['Numtppd', 'Numtpbi', 'Indtppd', 'Indtpbi']

    @staticmethod
    def transform_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
      logger.debug("Transform dataset..")
      dataset[DataPreparation.TARGET_COLUMN] = dataset['Numtppd'].apply(lambda x: 1 if x != 0 else 0)
      dataset = dataset.drop(columns=DataPreparation.NUMERICAL_COLUMNS)
      dataset = pd.get_dummies(dataset, columns=DataPreparation.CATEGORICAL_COLUMNS)
      return dataset

    @staticmethod
    def train_test_split(dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
      logger.debug(f"Split dataset into train-test with params: test_size={test_size}, random_state={random_state}")
      X = dataset.drop(DataPreparation.TARGET_COLUMN, axis=1)
      y = dataset[DataPreparation.TARGET_COLUMN]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
      return X_train, X_test, y_train, y_test