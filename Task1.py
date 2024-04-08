import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import itertools
import datetime
import os

import numpy as np
from scipy.fft import fft
from sklearn.metrics import mean_absolute_error
import warnings

TIMESTMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = '.\\output_task1_' + TIMESTMP + '\\'
os.mkdir(OUTPUT_DIR)
log_file_name = open(OUTPUT_DIR + f"\\Task1_log_{TIMESTMP}.txt", 'a')
DAYS_TO_PREDICT = 110
warnings.filterwarnings('ignore')


########################################################################################################################
"""
    Logs the passed data by writing it to an already opened log file and printing it to the console.

    Parameters:
    - data (str): The message or data to be logged.

    Returns:
    None.

"""
def logData(data):
    log_file_name.write(data)
    print(data)

########################################################################################################################
"""
    Visualizes a time series for the specified DataFrame column. This method creates a line plot for the selected
    column data against its corresponding date, providing insights into trends, patterns, or anomalies over time.

    Parameters:
    - myData (DataFrame): The DataFrame containing the time series data to be plotted. It must include a 'Date'
      column and the column specified by myCol.
    - myCol (str): The name of the column in myData to be visualized as a time series. This column should contain
      numerical data.
    - title (str): A title for the plot. This title is also used as part of the filename for saving the plot image.

    Returns:
    None.

    Notes:
    - The plot is saved to a file in the OUTPUT_DIR directory. The filename is constructed using the provided title
      and myCol name

"""
def visualizeTimeSeries(myData,myCol, title):
    sns.set(style="whitegrid")  

    plt.figure(figsize=(12, 6))
    plt.plot(myData['Date'], myData[myCol], color='green', linewidth=1, marker='o', markersize=4)

    plt.title(f'{title} Time Series Visualization of {myCol}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(myCol, fontsize=12)

    plt.xticks(rotation=45) 
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title}_{myCol}_timeSeries.png'))

    plt.show()
    

########################################################################################################################
"""
    Analyzes the time steps in the given dataset to determine the continuity and identify any gaps within the date range. 

    Parameters:
    - data (DataFrame): The dataset containing a 'Day' column, which can either be dates in datetime format or 
      sequential numbers representing days.

    Returns:
    - summary (dict): A dictionary containing the start day/date, end day/date, expected number of days, 
      actual number of days, and the calculated gap in measurement.

"""
def describeTimeSteps(data):
    # Check if 'Day' is already in a datetime format or a sequential number
    if pd.api.types.is_datetime64_any_dtype(data['Day']):
        # 'Day' is in datetime format
        data_sorted = data.sort_values(by='Day')
        start_date = data_sorted['Day'].iloc[0]
        end_date = data_sorted['Day'].iloc[-1]
        expected_days = pd.date_range(start=start_date, end=end_date).shape[0]
        actual_days = data_sorted.shape[0]
        gaps = expected_days - actual_days
        summary = {
            'Start Day': start_date,
            'End Day': end_date,
            'Expected Number of Days': expected_days,
            'Actual Number of Days': actual_days,
            'Gaps in Measurement': gaps
            }
    else:
        # 'Day' is a sequential number
        data_sorted = data.sort_values(by='Day')
        start_day = data_sorted['Day'].iloc[0]
        end_day = data_sorted['Day'].iloc[-1]
        expected_days = end_day - start_day + 1
        actual_days = data_sorted.shape[0]
        gaps = expected_days - actual_days
        summary = {
            'Start Day': start_day,
            'End Day': end_day,
            'Expected Number of Days': expected_days,
            'Actual Number of Days': actual_days,
            'Gaps in Measurement': gaps
            }

    return summary    

########################################################################################################################
"""
    Decomposes a time series into its observed, trend, seasonal, and residual components, and plots each component. 

    Parameters:
    - timeseries (Series): The time series data to be decomposed and plotted.
    - model (str, optional): The type of seasonal decomposition to perform. Can be either 'additive' or 'multiplicative'. 
      Default is 'additive'.
    - period (int, optional): The period of the seasonality to use in the decomposition. If not specified, the period 
      will be automatically inferred if possible.

    Returns:
    - decomposition (DecomposeResult): The result of the seasonal decomposition, containing the observed, trend, 
      seasonal, and residual components.

"""
def plotSeasonality(timeseries, model='additive', period=None):
    decomposition = seasonal_decompose(timeseries, model=model, period=period)

    sns.set(style="whitegrid")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    
    decomposition.observed.plot(ax=ax1, title='Observed', color='blue')
    ax1.set_ylabel('Observed')

    decomposition.trend.plot(ax=ax2, title='Trend', color='green')
    ax2.set_ylabel('Trend')

    decomposition.seasonal.plot(ax=ax3, title='Seasonal', color='purple')
    ax3.set_ylabel('Seasonal')

    decomposition.resid.plot(ax=ax4, title='Residual', color='red')
    ax4.set_ylabel('Residual')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'seasonality.png'))

    plt.show()

    return decomposition

########################################################################################################################
"""
    Evaluates the stationarity of a time series using the Augmented Dickey-Fuller (ADF) test and logs the results.

    Parameters:
    - timeseries (Series): The time series data to be tested for stationarity.
    - title (str): The title or description of the time series, used for logging purposes.
    - significance_level (float, optional): The significance level used to determine stationarity. Defaults to 0.05.

    Returns:
    - adf_output (dict): A dictionary containing the results of the ADF test, including the test statistic, p-value, 
      number of lags used, number of observations used, and critical values.

"""
def evaluateStationarity(timeseries, title,significance_level=0.05):
    # Perform the Augmented Dickey-Fuller test
    adf_test = adfuller(timeseries, autolag='AIC')
    adf_output = {
        'Test Statistic': adf_test[0],
        'p-value': adf_test[1],
        'Lags Used': adf_test[2],
        'Number of Observations Used': adf_test[3],
        'Critical Values': adf_test[4]
    }

    # Determine if the series is stationary based on the p-value
    is_stationary = adf_output['p-value'] < significance_level

    logData(f'{title} Results of Dickey-Fuller Test:')
    for key, value in adf_output.items():
        logData(f'   {key}: {value}')
    logData(f'\nIs the time series stationary? {"Yes" if is_stationary else "No"}\n')

    return adf_output

########################################################################################################################
"""
    Identifies the first significant lag in the Autocorrelation Function (ACF) which may suggest a seasonal period in 
    a given time series data. 

    Parameters:
    - timeseries (Series): The time series data for which to find the seasonal period.
    - max_lag (int, optional): The maximum lag to consider when computing the autocorrelation. Defaults to 30.
    - alpha (float, optional): The significance level used for the confidence interval. Defaults to 0.05, 
      representing a 95% confidence level.

    Returns:
    - int or None: The first significant lag if any are found, otherwise None. A significant lag is identified
      if the ACF value is outside the confidence interval.

"""
def findSeasonalPeriod(timeseries, max_lag=30, alpha=0.05):
    acf_vals = acf(timeseries, nlags=max_lag, alpha=alpha)

    # Find significant lags
    significant_lags = np.where(acf_vals[1][:, 1] > acf_vals[0])[0]

    # Apply Seaborn style
    sns.set(style="whitegrid")

    # Plot ACF
    plt.figure(figsize=(10, 6))
    stem_lines = plt.stem(range(max_lag+1), acf_vals[0], basefmt=" ")

    # Customize colors if all points are significant
    if len(significant_lags) == max_lag + 1:
        plt.setp(stem_lines, 'color', 'green', 'linewidth', 1)
        plt.title('All Lags are Significant in ACF', fontsize=14, color='green')
    else:
        plt.setp(stem_lines, 'color', 'green', 'linewidth', 1)
        # Highlight significant lags
        for lag in significant_lags:
            plt.plot(lag, acf_vals[0][lag], marker='o', markersize=2, color='black')

    plt.title('Autocorrelation Function (ACF) for Seasonality Detection', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('ACF Value', fontsize=12)

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'ACF1.png'))

    plt.show()

    # Return the first significant lag as potential seasonal period
    return significant_lags[0] if len(significant_lags) > 0 else None


########################################################################################################################
"""
    Analyzes and plots the spectral density of a given time series to identify dominant frequencies
    which can indicate seasonality or cyclic patterns. 

    Parameters:
    - myData (DataFrame): The dataset containing the time series data. It should have a 'Revenue' column.
    - title (str): The title for the plot and the prefix for the name of the saved plot file.

    Returns:
    None. The function saves the spectral density plot to a file and displays it inline.

"""
def analyzeSpectralDensity(myData, title):
    # Assuming 'Revenue' is the time series column
    ts_data = myData['Revenue']

    # Apply Fast Fourier Transform (FFT)
    fft_result = fft( np.array(ts_data))
    
    # Get power spectrum density (magnitude of the fft)
    psd = np.abs(fft_result)
    
    # Frequency bins (assuming equal spacing in time series)
    freq = np.fft.fftfreq(len(psd))

    # Apply Seaborn style
    sns.set(style="whitegrid")

    # Plot the Spectral Density
    plt.figure(figsize=(12, 6))
    plt.plot(freq, psd, color='green', linewidth=1)

    # Customize the plot
    plt.title(f'{title} Spectral Density of Revenue Time Series', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power Spectrum Density', fontsize=12)

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title}_spectralDensity.png'))

    plt.show()


########################################################################################################################
"""
    Evaluates the performance of ARIMA model predictions over multiple forecast horizons.

    This function fits an ARIMA model to the provided training data, then forecasts values over 
    increasing horizons from 1 up to `max_horizon`. It calculates the mean absolute error of these 
    forecasts against the actual test data for each horizon, plots these errors, and identifies the 
    horizon with the minimum error.

    Parameters:
    - train (array-like): The training dataset. Must be suitable for use with the ARIMA model.
    - test (array-like): The test dataset against which forecast accuracy is evaluated.
    - max_horizon (int): The maximum forecast horizon to evaluate. This function will evaluate all
                         horizons from 1 to `max_horizon` inclusive.
    - order (tuple of int, optional): The order (p,d,q) of the ARIMA model to be fitted. Defaults to (1, 1, 1).

    Outputs:
    - A plot saved to a predefined output directory showing the mean absolute error of forecasts at 
      each horizon, with an annotation for the horizon with the minimum error. The plot is also 
      displayed in the current notebook or script output.

"""
def evaluatePredictionHorizons(train, test, max_horizon, order=(1, 1, 1)):   
    # Fit the ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    errors = []

    # Forecast for different horizons and calculate errors
    for t in range(1, max_horizon + 1):
        forecast = model_fit.forecast(steps=t)
        actual = test[:t]
        error = mean_absolute_error(actual, forecast)
        errors.append(error)

    # Apply Seaborn style
    sns.set(style="whitegrid")

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_horizon + 1), errors, marker='o', color='deepskyblue', linewidth=2, markersize=6)

    # Customizing the plot
    plt.title('Mean Absolute Error over Different Forecast Horizons', fontsize=14, fontweight='bold')
    plt.xlabel('Forecast Horizon', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)

    # Optional: Annotate the minimum error
    min_error_horizon = errors.index(min(errors)) + 1
    plt.axvline(x=min_error_horizon, color='red', linestyle='--', label=f'Minimum Error at Horizon {min_error_horizon}')
    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictorHorizons.png'))

    plt.show()


########################################################################################################################
"""
    Transforms a time series into a stationary time series by differencing.

    This function makes a time series stationary by applying differencing of a specified order.
    Stationarity is a crucial assumption in many time series analysis methods, making this transformation
    an essential preprocessing step. The differencing process subtracts the current value of the series
    from the previous value, potentially over a specified number of lag periods (`order`). The first `order`
    rows, which result in NaN values due to the differencing, are dropped from the dataset.

    Parameters:
    - data (DataFrame): The original DataFrame containing the time series data.
    - column_name (str): The name of the column in `data` that contains the time series to be made stationary.
    - order (int, optional): The differencing order to apply. This determines the lag between the current
                             value and the value to subtract from it. Defaults to 1, which means that each
                             value in the series will be replaced with the difference from its immediate predecessor.

    Returns:
    - DataFrame: A DataFrame with the same structure as `data` but with the specified column transformed to
                 make the time series stationary. Rows resulting in NaN values from the differencing process
                 are dropped.

    Example:
    ```
    # Make the 'sales' column of a DataFrame stationary
    stationary_sales = makeTimeSeriesStationary(df, 'sales')
    ```
"""
def makeTimeSeriesStationary(data, column_name, order=1):
    # Copy the original data to avoid modifying it
    stationary_data = data.copy()
    
    # Perform differencing
    stationary_data[column_name] = stationary_data[column_name].diff(periods=order)
    
    # Drop NaN values resulting from differencing
    stationary_data.dropna(inplace=True)
    
    return stationary_data

########################################################################################################################
"""
    Identifies the best (p, d, q) parameters for an ARIMA model based on the Akaike Information Criterion (AIC).

    This function iterates over combinations of p, d, q parameters within a predefined range (0 to 4 for each)
    and fits an ARIMA model to the provided training data. It evaluates each model's goodness of fit using the AIC,
    selecting the combination of parameters that results in the lowest AIC value. The AIC aims to balance model 
    complexity against the quality of the fit, with lower values indicating a model that fits well without being 
    overly complex. This function is particularly useful for time series analysis where the optimal order of 
    differencing and autoregressive and moving average components is not known in advance.

    Parameters:
    - myTrain (DataFrame): A DataFrame containing the training dataset. This dataset must include a column named 'Revenue'
                           which contains the time series data to model.

    Returns:
    - tuple: The best (p, d, q) parameters for the ARIMA model as determined by the lowest AIC.

    Outputs:
    - Prints the list of all (p, d, q) combinations evaluated and the best combination found along with its AIC.
"""
def bestPDQ(myTrain):
    p = d = q = range(0, 5)  # Define the range for p, d, q
    pdq_combinations = list(itertools.product(p, d, q))  

    best_aic = float('inf')
    best_pdq = None
    best_model = None

    print(pdq_combinations)

    for pdq in pdq_combinations:
        try:
            temp_model = ARIMA(myTrain['Revenue'], order=pdq)
            temp_model_fit = temp_model.fit()
            if temp_model_fit.aic < best_aic:
                best_aic = temp_model_fit.aic
                best_pdq = pdq
                best_model = temp_model_fit
        except:
            continue
        
    print(f'Best ARIMA Model: {best_pdq} with AIC: {best_aic}')
        
    return best_pdq

########################################################################################################################
########################################################################################################################
# Note the start time so we can monitor overall performance
startTime = datetime.datetime.now()

# 1. Load the data ############################################################
data = pd.read_csv('teleco_time_series.csv')

# Drop any rows with null values
data = data.dropna()
data = data.reset_index(drop=True)

# Convert the index to a proper date
data['Date'] = (pd.date_range(start=datetime.datetime(2019,1,1),periods=data.shape[0],freq='24H'))


# 2. Analyze the original data ################################################

# Visualize the unaltered data
visualizeTimeSeries(data,'Revenue', 'initial')


# describe the data
time_step_summary = describeTimeSteps(data)
logData('\n')
logData(str(time_step_summary))
logData('\n')

# Evaluate the Stationarity of the original data
adf_results = evaluateStationarity(data['Revenue'],'Initial')

# Plot the decomposition of the of the original data
seasonal_decomposition = plotSeasonality(data['Revenue'], period=365)

# Find the Seasonality period
seasonal_period = findSeasonalPeriod(data['Revenue'], max_lag=365)
print("Suggested Seasonal Period:", seasonal_period, '\n')


# 3. Data Preparation Steps ###################################################

# Make the data Stationary
stationary_data = makeTimeSeriesStationary(data, 'Revenue')

# Visualize the Stationary TimeSeries
visualizeTimeSeries(stationary_data,'Revenue', 'Stationary')

# evaluate the Stationarity again to confirm
adf_results = evaluateStationarity(stationary_data['Revenue'],'Final')

# Plot the decomposition of the stationary data
seasonal_decomposition = plotSeasonality(stationary_data['Revenue'], period=365)

# Plot the spectral density of the stationary data 
analyzeSpectralDensity(stationary_data,'Final')

# Save the Dataset
stationary_data.to_csv('cleaned_teleco_time_series.csv', index=False)


# 4.  Model Identification and Analysis

# Plot the ACF and Partial ACF
plot_acf(stationary_data['Revenue'], lags=30)
plt.savefig(os.path.join(OUTPUT_DIR, 'ACF2.png'))

plot_pacf(stationary_data['Revenue'], lags=30)
plt.savefig(os.path.join(OUTPUT_DIR, 'PACF.png'))


# Identify and train an ARIMA Model

# Split the data into test and training
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]
logData("Data split into training and test sets.")

# Find the best P,D,Q params to use for the model
best_pdq = bestPDQ(train)


# Plot the MAE for different prediction horizons so we can pick the best one
evaluatePredictionHorizons(train['Revenue'], test['Revenue'], max_horizon=len(test), order=(best_pdq))


# Train a model with the best P,D,Q params
model = ARIMA(train['Revenue'], order=(best_pdq))
model_fit = model.fit()

print(model_fit.summary())


# 5. Use the model to Forecast DAYS_TO_PREDICT ahead
forecast_result = model_fit.get_forecast(steps=DAYS_TO_PREDICT)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence intervals

# Create a date range for the forecast period
foreIndex = pd.RangeIndex(start=test.index[-1], stop=test.index[-1]+DAYS_TO_PREDICT, step=1)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Revenue'], label='Training Data')
plt.plot(test.index, test['Revenue'], label='Test Data')
plt.plot(foreIndex, forecast, label='Forecasted Data', color='red')

# Plot the Confidence Intervals
plt.fill_between(foreIndex, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')

plt.title(f'{DAYS_TO_PREDICT}-Day Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()


plt.savefig(os.path.join(OUTPUT_DIR, f'final.png'))
plt.show()

# Note the stop time and compute the runtime
stopTime = datetime.datetime.now()
elapsedTime = stopTime - startTime
logData('\n Elapsed Time: ' + str(elapsedTime))