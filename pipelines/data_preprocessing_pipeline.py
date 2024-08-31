import pandas as pd
import logging


class DataPreprocessingPipeline:
    def __init__(self, df):
        """
        Initialize the DataPreprocessingPipeline with a DataFrame.

        :param df: DataFrame containing the time series data with columns 'ds' and 'y'.
        """
        self.df = df
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def process(self):
        """
        Main method to process the DataFrame by handling missing values,
        outliers, and ensuring data consistency.

        :return: Processed DataFrame or None if an error occurs, and an error message.
        """
        try:
            self.validate_data()
            self.df['ds'] = pd.to_datetime(self.df['ds'])
            self.handle_missing_values()
            self.df = self.handle_outliers(self.df, column='y')
            self.logger.info("Data preprocessing completed successfully.")
            return self.df, None

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            return None, str(e)

    def validate_data(self):
        """
        Validate the DataFrame structure and types.
        Raises an exception if validation fails.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        required_columns = ['ds', 'y']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        if not pd.api.types.is_datetime64_any_dtype(self.df['ds']):
            raise ValueError("Column 'ds' must be of datetime type.")

        if not pd.api.types.is_numeric_dtype(self.df['y']):
            raise ValueError("Column 'y' must be numeric.")

        self.logger.info("Data validation passed.")

    def handle_missing_values(self):
        """
        Handle missing values in the DataFrame.
        The method uses interpolation, forward fill, and backward fill to handle NaNs.
        """
        initial_missing = self.df['y'].isna().sum()
        if initial_missing > 0:
            self.logger.info(f"Handling {initial_missing} missing values in 'y' column.")
            self.df['y'] = self.df['y'].interpolate(method='linear').ffill().bfill()
            final_missing = self.df['y'].isna().sum()
            self.logger.info(f"Missing values after processing: {final_missing}")
        else:
            self.logger.info("No missing values found in 'y' column.")

    def handle_outliers(self, df, column, method='zscore', threshold=3):
        """
        Handle outliers in the specified column using the chosen method.

        :param df: DataFrame to process.
        :param column: Column name where outliers should be handled.
        :param method: Method to use for outlier detection ('zscore', 'iqr').
        :param threshold: Threshold to use for detecting outliers.
        :return: DataFrame with outliers handled.
        """
        df = df.copy()

        if method == 'zscore':
            df['z_score'] = (df[column] - df[column].mean()) / df[column].std()
            outliers = df[df['z_score'].abs() > threshold]
            self.logger.info(f"Detected {len(outliers)} outliers using z-score method.")
            df = df[df['z_score'].abs() <= threshold]
            df.drop(columns=['z_score'], inplace=True)

        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            self.logger.info(f"Detected {len(outliers)} outliers using IQR method.")
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        else:
            raise ValueError(f"Outlier detection method '{method}' is not supported.")

        self.logger.info(f"Outliers removed. Remaining data points: {len(df)}")
        return df
