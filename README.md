# Stock Price Prediction Application

## Overview

This Stock Price Prediction Application allows users to predict future stock prices using historical data. The app leverages machine learning models, including Facebook Prophet, to forecast stock prices based on past trends. Users can either upload their own CSV files or download data directly from Yahoo Finance, and the app provides options to train a new model, retrain an existing model, or use a pre-trained model.

## Features

- **Data Source Options**: Download data directly from Yahoo Finance using a stock symbol or upload your own CSV file for training and prediction.
  
- **Model Options**: Use a pre-trained model, retrain a pre-trained model, or train a new model from scratch.

- **Prediction Options**: Set prediction intervals for confidence levels, include or exclude holidays in the model, and choose between additive or multiplicative seasonality.

- **Results**: Visualize the comparison between training and test data, download the test dataset used for evaluation, view detailed metrics such as MSE, RMSE, MAE, and MAPE, and see actual vs. predicted values for the uploaded test data.

## Project Structure

- **app.py**: Main Flask application file
- **pipelines/**: Directory containing the pipeline modules
  - `__init__.py`: Init file for the pipelines module
  - **data_preprocessing_pipeline.py**: Data preprocessing logic
  - **dataset_creation_pipeline.py**: Dataset creation and splitting logic
  - **model_pipeline.py**: Model training, retraining, and prediction logic
- **static/**: Directory for static assets
  - **css/**: Custom styles for the application
  - **images/**: Placeholder or generated graphs
  - **data/**: Sample CSV data for download
- **templates/**: Directory containing the HTML templates
  - **index.html**: Main HTML template for the application
- **README.md**: Project README file

## Installation

1. **Prerequisites**: Ensure you have Python 3.7 or higher, pip, and virtualenv installed.
2. **Clone the repository**: 
   ```
   git clone https://github.com/yourusername/stock-price-prediction.git
   ```
3. **Navigate to the project directory**: 
   ```
   cd stock-price-prediction
   ```
4. **Set up the environment**:
   - Create a virtual environment: 
     ```
     python -m venv .venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```
       .venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```
       source .venv/bin/activate
       ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
5. **Run the Application**: Start the Flask application with 
   ```
   python app.py
   ```
   and access it via `http://127.0.0.1:5000/` in your web browser.

## Usage

1. **Select Data Source**: Choose `yfinance` for Yahoo Finance data or `Upload CSV` for your own dataset.
2. **Configure Prediction Settings**: Set the start and end dates, choose seasonality mode, and specify prediction interval width.
3. **Choose a Model Option**: Use, retrain, or train a new model.
4. **View Results**: After prediction, view the metrics, graphs, and predictions.
5. **Test New Data**: Upload a new dataset for prediction and view the results.

## Screenshots

Include relevant screenshots of the application in action.

## Contributing

1. Fork the repository
2. Create a new feature branch
   ```
   git checkout -b feature-branch-name
   ```
3. Commit your changes
   ```
   git commit -m "Your detailed description of the changes"
   ```
4. Push to the branch
   ```
   git push origin feature-branch-name
   ```
5. Open a pull request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

