import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetCreationPipeline:
    def __init__(self, processed_data):
        """
        Initialize the DatasetCreationPipeline with preprocessed data.

        :param processed_data: DataFrame containing the processed time series data.
        """
        self.processed_data = processed_data

    def create_dataset(self, test_size=0.2, validation_size=0.1):
        """
        Create training, validation, and test datasets from the processed data.

        :param test_size: Proportion of the data to be used as the test set.
        :param validation_size: Proportion of the training data to be used as the validation set.
        :return: Tuple of (train_data, validation_data, test_data, error_message).
                 - train_data: DataFrame for training the model.
                 - validation_data: DataFrame for validating the model.
                 - test_data: DataFrame for testing the model.
                 - error_message: None if successful, otherwise a string describing the error.
        """
        try:
            # Ensure the processed data is valid
            if not self.validate_data(self.processed_data):
                return None, None, None, "Invalid data: Ensure the dataset is properly formatted and preprocessed."

            # Split the data into training + validation and test sets
            train_val_data, test_data = train_test_split(
                self.processed_data, test_size=test_size, shuffle=False
            )

            # Further split the training data into training and validation sets
            train_data, validation_data = train_test_split(
                train_val_data, test_size=validation_size, shuffle=False
            )

            return train_data, validation_data, test_data, None

        except Exception as e:
            return None, None, None, str(e)

    def validate_data(self, data):
        """
        Validate the processed data for fbprophet compatibility.

        :param data: DataFrame to validate.
        :return: Boolean indicating whether the data is valid.
        """
        required_columns = ['ds', 'y']

        # Check for required columns
        if not all(column in data.columns for column in required_columns):
            return False

        # Check for missing values
        if data.isnull().any().any():
            return False

        # Check if 'ds' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(data['ds']):
            return False

        # Check if 'y' is numeric
        if not pd.api.types.is_numeric_dtype(data['y']):
            return False

        return True
