#Dependencies 
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from tpot import TPOTRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer, mean_absolute_error
from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings('ignore')

def prepare_stock_data(df, prediction_window=5):
    """
    Purpose: 
    This function prepares stock data for predictive modeling by performing several preprocessing steps.

    Parameters:
    df: A pandas DataFrame containing stock data with columns like 'Date', 'Stock_Symbol', 'Stock_Close', and 'Stock_Volume'.
    prediction_window (optional): An integer representing the number of days to look ahead for predicting future stock prices. Default is 5.
   
    Steps:
    Converts the 'Date' column to datetime format.
    Sorts the data by 'Stock_Symbol' and 'Date'.
    Calculates technical indicators such as stock returns, volume changes, moving averages, and volatility.
    Creates lagged features for sentiment analysis and closing prices.
    Computes future target variables like next week's closing price and its returns.
    Handles missing values using KNN imputation.
    
    Returns:
    A DataFrame with preprocessed stock data ready for modeling.

    Misc:
    Creating lagged features and moving averages in stock data analysis serves several important purposes.
    It allows the model to learn from past values and make predictions based on historical patterns.

    Temporal Dependencies: Financial time series, like stock prices, often exhibit temporal dependencies where past values influence future behavior.
    Lagged features help capture these dependencies by providing historical context to the model.

    Noise Reduction: By incorporating lagged values, analysts can smooth out short-term fluctuations and focus on longer-term trends,
    which are often more predictive of future movements than recent noise.

    Predictive Modeling: Including moving averages as features in predictive models helps capture the momentum effect,
    where past price movements influence future price changes. This is particularly useful for forecasting models that aim to predict stock returns.

    """
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Stock_Symbol', 'Date'])
    
    # Create technical indicators
    df['Stock_Returns'] = df.groupby('Stock_Symbol')['Stock_Close'].pct_change()
    df['Stock_Volume_Change'] = df.groupby('Stock_Symbol')['Stock_Volume'].pct_change()

    # Handle division by zero in returns and volume change
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Create moving averages
    df['Stock_MA5'] = df.groupby('Stock_Symbol')['Stock_Close'].transform(lambda x: x.rolling(window=5).mean())
    df['Stock_MA20'] = df.groupby('Stock_Symbol')['Stock_Close'].transform(lambda x: x.rolling(window=20).mean())

    # Create volatility measure
    df['Stock_Volatility'] = df.groupby('Stock_Symbol')['Stock_Returns'].transform(lambda x: x.rolling(window=20).std())

    # Create lagged features
    sentiment_cols = [col for col in df.columns if 'sentiment' in col]
    for col in sentiment_cols:
        df[f'{col}_lag1'] = df.groupby('Stock_Symbol')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('Stock_Symbol')[col].shift(2)

    # Create price lags
    df['Stock_Close_lag1'] = df.groupby('Stock_Symbol')['Stock_Close'].shift(1)
    df['Stock_Close_lag2'] = df.groupby('Stock_Symbol')['Stock_Close'].shift(2)

    # Create target variable (next week's closing price)
    df['Stock_Future_Close'] = df.groupby('Stock_Symbol')['Stock_Close'].shift(-prediction_window)
    df['Stock_Future_Returns'] = (df['Stock_Future_Close'] - df['Stock_Close']) / df['Stock_Close']

    # Handle division by zero in index future returns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Specify the column you want to ignore as a list of strings
    ignore_columns = ["Date", "Stock_Symbol",]

    # Split the DataFrame into columns to impute and the ones to ignore
    columns_to_impute = df.drop(columns=ignore_columns)
    ignored_column = df.loc[:, ignore_columns]

    # Perform imputation on the selected columns only
    imputer = KNNImputer(n_neighbors=5)
    df_imputed_columns = pd.DataFrame(imputer.fit_transform(columns_to_impute), 
                                    columns=columns_to_impute.columns, 
                                    index=columns_to_impute.index)

    # Reattach the ignored column(s)
    df_imputed = df_imputed_columns.join(ignored_column)

    print("Addition of lagged features and future returns completed")

    return df_imputed

def train_predict_model(df, stock , test_size=0.2, prediction_window=5):
    """
    Purpose:
    Trains a predictive model for a specific stock using the TPOT library and evaluates its performance.

    Parameters:
    df: A pandas DataFrame containing preprocessed stock data.
    stock: A string representing the stock symbol for which the model is to be trained.
    test_size (optional): A float between 0.0 and 1.0 indicating the proportion of the dataset to include in the test split. Default is 0.2.
    prediction_window (optional): An integer representing the number of days ahead for prediction. Default is 5.
    
    Steps:
    Prepares the stock data using prepare_stock_data.
    Splits the data into training and testing sets.
    Scales the features using standard scaling.
    Defines a custom scoring function to evaluate model performance.
    Initializes and trains a TPOT regressor model.
    Exports the best-performing pipeline as a Python script.
    Makes predictions on the test set and calculates evaluation metrics.
    
    Returns:
    A dictionary containing model results, including metrics, predictions, actual prices, and the trained pipeline for each stock.

    """
    results = {}
    
    # Prepare data
    df = prepare_stock_data(df, prediction_window=prediction_window)
    stock_df = df[df['Stock_Symbol'] == stock].copy()
    
    # Split into features and target
    X, y = stock_df.drop(columns=['Date', 'Stock_Symbol']), stock_df['Stock_Close']
    
    # Create time series split
    split_idx = int(len(stock_df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print("Data split into training and testing sets")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed")

        
    # Define custom scorer that balances absolute error with direction accuracy
    def custom_stock_scorer(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        direction_accuracy = np.mean((np.diff(y_true) * np.diff(y_pred)) > 0)
        return -mae * (1 + direction_accuracy)  # Negative because TPOT maximizes


    print('Intalizing TPOT Regressor')
    tpot_model = TPOTRegressor(
        generations=7, 
        population_size=75, 
        verbosity=2,
        random_state=42, 
        scoring=make_scorer(custom_stock_scorer),
        n_jobs=-2, 
        max_time_mins=10,
    )
    
    tpot_model.fit(X_train_scaled, y_train)

    #Exporting the pipeline
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    tpot_model.export(f'{stock}_{current_time}_prediction_.py')

    # Make predictions
    y_pred = tpot_model.predict(X_test_scaled)

    # No need to convert returns to prices since we're predicting prices directly
    test_dates = stock_df.iloc[split_idx:]['Date']
    actual_prices = stock_df.iloc[split_idx:]['Stock_Close']
    predicted_prices = y_pred  # Direct price predictions

    # Calculate metrics directly with prices
    mse = mean_squared_error(actual_prices, predicted_prices)
    mape = mean_absolute_percentage_error(actual_prices[prediction_window:], 
                                        predicted_prices[:-prediction_window])
    r2 = r2_score(actual_prices[prediction_window:], predicted_prices[:-prediction_window])

    # Feature importance extraction
    pipeline = tpot_model.fitted_pipeline_

    results[stock] = {
        'model': tpot_model,
        'predictions': predicted_prices,
        'actual': actual_prices,
        'mse': mse,
        'mape': mape,
        'r2': r2,
        'test_dates': test_dates,
        'pipeline': pipeline
    }

    return results

def display_results(model_results):
    """
    Purpose:
    Displays the results of model training and prediction in a readable format.

    Parameters:
    model_results: A dictionary where keys are stock symbols and values are dictionaries containing model performance metrics, predictions, actual prices, etc.
    
    Steps:
    Iterates over each stock's results.
    Prints performance metrics like MSE, MAPE, and R² score.
    Plots a line chart comparing actual vs. predicted stock prices using Plotly Express.
    Displays a DataFrame with test dates, actual prices, predicted prices, and price differences.

    Returns: None (The function is used for displaying results only).

    """
    for stock, result in model_results.items():
        print(f"\nResults for Stock: {stock}")
       
        # Performance Metrics
        print("\nPerformance Metrics:")
        print(f" Mean Squared Error (MSE): {result['mse']:.4f}")
        print(f" Mean Absolute Percentage Error (MAPE): {result['mape']:.2%}")
        print(f" R² Score: {result['r2']:.4f}\n")

        # Test Dates, Actual Prices and Predicted Prices
        test_dates = result['test_dates']
        actual_prices = result['actual']
        predicted_prices = result['predictions']
        price_diff = actual_prices-predicted_prices
        comparison_df = pd.DataFrame({
            'Date': test_dates,
            'Actual Price': actual_prices,
            'Predicted Price': predicted_prices,
            'Price Difference': price_diff
        })

        # Plotly Express plot for Actual vs Predicted Prices
        fig = px.line(comparison_df, x='Date', y=['Actual Price', 'Predicted Price'],
                      labels={'value': 'Price', 'variable': 'Type'},
                      title=f'Actual vs. Predicted Stock Prices for {stock}')

        fig.update_traces(mode='lines+markers')
        fig.show()

        # Display the DataFrame with limited columns for readability
        print("\nTest Dates, Actual Prices and Predicted Prices:")
        print(comparison_df.head(10).to_string(index=False))

# Loading data
df = pd.read_excel("merged_sentiment_6months.xlsx")

# Drop columns with mostly empty values
df = df.dropna(axis=1, how='all')

unique_stocks = df['Stock_Symbol'].unique()
print("Pick from available stocks:")
for symbol in unique_stocks:
    print(symbol)

# Ensure valid input by looping until a correct stock symbol is provided
stock_symbol = None
while stock_symbol not in unique_stocks:
    user_input = input("Enter stock symbol: ").upper()
    if user_input in unique_stocks:
        stock_symbol = user_input
    else:
        print(f"Invalid stock symbol. Please choose from the following: {', '.join(unique_stocks)}")

print(f"You have selected the stock symbol: {stock_symbol}")

df = df[df['Stock_Symbol'] == stock_symbol].reset_index(drop=True)

# Call the train_predict_model function
model_results = train_predict_model(df, stock=stock_symbol)

# Display results using the display_results function
display_results(model_results)