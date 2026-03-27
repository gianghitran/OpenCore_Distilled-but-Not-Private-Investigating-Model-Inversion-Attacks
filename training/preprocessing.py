import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.pipeline import Pipeline


class Preprocessing:
    def __init__(self, output_dir="data"):
        """
        Initialize the Preprocessing class.
        Args:
            output_dir (str): The directory where preprocessed data will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def fit_transform(self, path: str):
        """
        Preprocess the data at the given path with Quantile Transformer and MinMax Scaler.
        Args:
            path (str): The path to the data file.
        Returns:
            tuple: A tuple containing the preprocessed data and the scaler.
        """
        
        # Load the data
        path = Path(path)
        df = pd.read_csv(path)


        # {..private...}
        
        
        # Save the preprocessed data
        base = path.stem        
        pd.DataFrame(X_train).assign(label=y_train).to_csv(
            self.output_dir / f"train_set_{base}.csv", index=False)
        pd.DataFrame(X_test).assign(label=y_test).to_csv(
            self.output_dir / f"test_set_{base}.csv", index=False)
        
        return X_train, X_test, y_train, y_test