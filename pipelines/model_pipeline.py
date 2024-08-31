from prophet import Prophet
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import numpy as np


class ModelPipeline:
    def __init__(self, train_data, seasonality_mode='additive', include_holidays=True, interval_width=0.8):
        """
        Initialize the ModelPipeline with the necessary data and parameters.

        :param train_data: DataFrame containing the training data.
        :param seasonality_mode: Seasonality mode to be used by Prophet ('additive' or 'multiplicative').
        :param include_holidays: Boolean indicating whether to include holidays in the model.
        :param interval_width: Interval width for uncertainty intervals.
        """
        self.train_data = train_data
        self.seasonality_mode = seasonality_mode
        self.include_holidays = include_holidays
        self.interval_width = interval_width
        self.model = None
        self.model_path = 'models/prophet_model.pkl'

        # Ensure the models and static/images directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('static/images', exist_ok=True)

    def load_pretrained_model(self):
        """Load the pretrained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def save_model(self):
        """Save the trained model to disk."""
        joblib.dump(self.model, self.model_path)

    def train_model(self):
        """Train a new model or retrain an existing model."""
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=self.get_holidays() if self.include_holidays else None,
            interval_width=self.interval_width
        )
        self.model.fit(self.train_data)

    def get_holidays(self):
        """Define holidays to include in the model."""
        # Example: You can define custom holidays here or use predefined ones.
        holidays = None
        return holidays

    def predict(self, model_option='pretrained'):
        """
        Make predictions based on the model option selected.

        :param model_option: Option to use ('pretrained', 'retrain', 'new').
        :return: Tuple containing predictions, metrics, graph paths, and an error message if any.
        """
        try:
            if model_option == 'pretrained':
                if not self.load_pretrained_model():
                    return None, None, None, "No pretrained model found. Please train a new model first."

            if model_option == 'retrain':
                if not self.load_pretrained_model():
                    return None, None, None, "No pretrained model found. Please train a new model first."
                self.train_model()
                self.save_model()

            if model_option == 'new':
                self.train_model()
                self.save_model()

            # Generate predictions
            future = self.model.make_future_dataframe(periods=365)  # Example: predict for 1 year ahead
            forecast = self.model.predict(future)

            # Calculate performance metrics
            metrics = self.calculate_metrics(forecast)

            # Save prediction graphs to disk
            graph_paths = self.save_graphs(forecast)

            return forecast, metrics, graph_paths, None

        except Exception as e:
            return None, None, None, str(e)

    def calculate_metrics(self, forecast):
        """
        Calculate performance metrics.

        :param forecast: Forecasted DataFrame from Prophet.
        :return: Dictionary containing calculated metrics.
        """
        y_true = self.train_data['y'].values
        y_pred = forecast['yhat'].iloc[:len(y_true)].values

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape':mape
        }

        # Save loss (MSE) over time for visualization
        self.save_loss_graph(mse)

        return metrics

    @staticmethod
    def save_loss_graph(mse):
        """
        Save a graph of the loss (MSE) over time.

        :param mse: Mean Squared Error of the model.
        """
        plt.figure(figsize=(10, 6))
        plt.plot([mse], marker='o', linestyle='-')
        plt.title('Model Loss Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.grid(True)
        graph_path = 'static/images/loss.png'
        plt.savefig(graph_path)
        plt.close()

    def save_graphs(self, forecast):
        """
        Save prediction graphs to disk, including a comparison of train vs. test data.

        :param forecast: Forecasted DataFrame from Prophet.
        :return: List of paths to the saved graphs.
        """
        graph_paths = []

        # Plot the forecast
        fig1 = self.model.plot(forecast)
        graph_path1 = 'static/images/forecast.png'
        fig1.savefig(graph_path1)
        graph_paths.append(graph_path1)

        # Plot components (trend, yearly seasonality, etc.)
        fig2 = self.model.plot_components(forecast)
        graph_path2 = 'static/images/forecast_components.png'
        fig2.savefig(graph_path2)
        graph_paths.append(graph_path2)

        # Plot comparison of train data vs. test data
        self.save_train_vs_test_graph(forecast)
        graph_paths.append('static/images/train_vs_test.png')

        plt.close('all')  # Close plots to free up memory

        return graph_paths

    def save_train_vs_test_graph(self, forecast):
        """
        Save a comparison graph of the training data vs. the test data.

        :param forecast: Forecasted DataFrame from Prophet.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_data['ds'], self.train_data['y'], label='Train Data')
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Test Data')
        plt.title('Training Data vs. Predicted Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        graph_path = 'static/images/train_vs_test.png'
        plt.savefig(graph_path)
        plt.close()
