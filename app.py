from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
import yfinance as yf
from pipelines.data_preprocessing_pipeline import DataPreprocessingPipeline
from pipelines.dataset_creation_pipeline import DatasetCreationPipeline
from pipelines.model_pipeline import ModelPipeline

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['DOWNLOAD_FOLDER'] = 'downloads/'

# Ensure the upload and download folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images/', exist_ok=True)


# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data_source = request.form.get('data_source')
        stock_symbol = request.form.get('stock_symbol')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        seasonality_mode = request.form.get('seasonality_mode')
        include_holidays = request.form.get('include_holidays') == 'yes'
        interval_width = float(request.form.get('interval_width'))
        model_option = request.form.get('model_option')

        if data_source == 'yfinance':
            try:
                # Download data from yfinance
                df = yf.download(stock_symbol, start=start_date, end=end_date)
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

                # Step 1: Data Preprocessing
                preprocessing_pipeline = DataPreprocessingPipeline(df)
                processed_data, error_message = preprocessing_pipeline.process()

                if error_message:
                    flash(f'Data Preprocessing Error: {error_message}', 'error')
                    return redirect(request.url)

            except Exception as e:
                flash(f'Error fetching data from yfinance: {str(e)}', 'error')
                return redirect(request.url)

        else:
            if 'csv_file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)

            file = request.files['csv_file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Convert CSV to DataFrame
                df = pd.read_csv(file_path)

                # Step 1: Data Preprocessing
                preprocessing_pipeline = DataPreprocessingPipeline(df)
                processed_data, error_message = preprocessing_pipeline.process()

                if error_message:
                    flash(f'Data Preprocessing Error: {error_message}', 'error')
                    return redirect(request.url)
            else:
                flash('Invalid file type. Please upload a CSV file.', 'error')
                return redirect(request.url)

        # Step 2: Dataset Creation
        dataset_pipeline = DatasetCreationPipeline(processed_data)
        train_data, validation_data, test_data, error_message = dataset_pipeline.create_dataset()

        if error_message:
            flash(f'Dataset Creation Error: {error_message}', 'error')
            return redirect(request.url)

        # Save test data for download
        test_data_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f'test_data_{stock_symbol}.csv')
        test_data.to_csv(test_data_path, index=False)

        # Step 3: Model Prediction
        model_pipeline = ModelPipeline(train_data, seasonality_mode=seasonality_mode,
                                       include_holidays=include_holidays, interval_width=interval_width)
        prediction_results, metrics, graph_paths, error_message = model_pipeline.predict(model_option=model_option)

        if error_message:
            flash(f'Model Prediction Error: {error_message}', 'error')
            return redirect(request.url)

        # Render the templates with results
        return render_template('index.html',
                               prediction_results=prediction_results,
                               test_data_url=url_for('download_test_data', filename=f'test_data_{stock_symbol}.csv'),
                               mse=metrics.get('mse'),
                               rmse=metrics.get('rmse'),
                               mae=metrics.get('mae'),
                               mape=metrics.get('mape'),
                               graph_paths=graph_paths)

    return render_template('index.html')


@app.route('/download_test_data/<filename>')
def download_test_data(filename):
    return send_file(os.path.join(app.config['DOWNLOAD_FOLDER'], filename), as_attachment=True)


@app.route('/test_new_data', methods=['POST'])
def test_new_data():
    if 'new_data_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['new_data_file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert CSV to DataFrame
        df = pd.read_csv(file_path)

        # Step 1: Data Preprocessing
        preprocessing_pipeline = DataPreprocessingPipeline(df)
        processed_data, error_message = preprocessing_pipeline.process()

        if error_message:
            flash(f'Data Preprocessing Error: {error_message}', 'error')
            return redirect(url_for('index'))

        # Step 2: Model Prediction
        model_pipeline = ModelPipeline(processed_data)
        test_data_predictions, error_message = model_pipeline.predict(
            model_option='pretrained')  # Assuming using pretrained model for test data

        if error_message:
            flash(f'Test Data Prediction Error: {error_message}', 'error')
            return redirect(url_for('index'))

        # Combine actual and predicted values for display
        combined_results = []
        for row, prediction in zip(processed_data.to_dict('records'), test_data_predictions):
            combined_results.append((row['ds'], row['y'], prediction))

        # Pass the combined results to the template
        return render_template('index.html', test_data_predictions=combined_results)

    flash('Invalid file type. Please upload a CSV file.', 'error')
    return redirect(url_for('index'))


@app.route('/download_sample')
def download_sample():
    return redirect(url_for('static', filename='data/sample.csv'))


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
