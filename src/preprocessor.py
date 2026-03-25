from types import SimpleNamespace

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self, df: DataFrame, config: SimpleNamespace):
        self.raw_data = df
        self.preprocessed_data = self.raw_data.copy()
        self.X, self.Y = None, None
        self.config = config

        # mappings for config
        self.categorical_missing_method_mapping = {
            'mode': lambda col: col.mode()[0] if not col.mode().empty else None,
        }
        self.numerical_missing_method_mapping = {
            'median': lambda col: col.median(),
            'mean': lambda col: col.mean(),
        }

        self.categorical_missing_method_pointer = self.categorical_missing_method_mapping[self.config.categorical_missing_method]
        self.numerical_missing_method_pointer = self.numerical_missing_method_mapping[self.config.numerical_missing_method]

    def preprocess(self) -> tuple[DataFrame, DataFrame]:
        self.handle_missing_values()
        self.handle_outliers()
        self.encode_non_numeric_data()
        self.split()
        return self.X, self.Y

    def handle_missing_values(self):
        for col in self.raw_data.columns:
            if self.raw_data[col].isnull().sum() > 0:
                if self.raw_data[col].dtype in ['object', 'category']:
                    val = self.categorical_missing_method_pointer(self.raw_data[col])
                    self.preprocessed_data[col].fillna(val, inplace=True)
                else:
                    val = self.numerical_missing_method_pointer(self.raw_data[col])
                    self.preprocessed_data[col].fillna(val, inplace=True)


    def handle_outliers(self):
        columns = self.raw_data.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.preprocessed_data[col].quantile(0.25)
            Q3 = self.preprocessed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR

            self.preprocessed_data[col] = self.preprocessed_data[col].clip(lower_bound, upper_bound)

    def encode_non_numeric_data(self):
        categorical_cols = self.preprocessed_data.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col == self.config.target_column or self.preprocessed_data[col].nunique() == 2:
                # binary data and target column: Label Encoding
                le = LabelEncoder()
                self.preprocessed_data[col] = le.fit_transform(self.preprocessed_data[col].astype(str))
            else:
                # multi-category data: one-hot encoding
                self.preprocessed_data = pd.get_dummies(
                    self.preprocessed_data,
                    columns=[col],
                    drop_first=True,
                    prefix=col
                )

    def split(self):
        self.Y = self.preprocessed_data[self.config.target_column]
        self.X = self.preprocessed_data.drop(self.config.target_column, axis=1)


    # def split_data(self):
    #     y = self.preprocessed_data[self.target_col]
    #     X = self.preprocessed_data.drop(self.target_col, axis=1)
    #
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         X,
    #         y,
    #         test_size=self.test_size,
    #         random_state=self.random_state
    #     )
    #
    # def scale_data(self):
    #     self.scaler.fit(self.X_train)
    #     self.X_train = self.scaler.transform(self.X_train)
    #     self.X_test = self.scaler.transform(self.X_test)



