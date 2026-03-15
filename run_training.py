"""
Script to run the model training notebook code
"""
import pandas as pd
import numpy as np
import os
# Matplotlib is optional - only needed for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Matplotlib not available - visualization will be skipped")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import warnings
import sys
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# Step 1: Setup folders and discover available data files
data_folder = "data"
models_folder = "models"

# Create models folder if it doesn't exist
os.makedirs(models_folder, exist_ok=True)

# Discover available CSV files in data folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print(f"\n📁 Found {len(csv_files)} CSV files in data folder:")
for csv_file in csv_files:
    print(f"   - {csv_file}")

# Extract stock names from CSV files
stock_names = [f.replace('_NS.csv', '') for f in csv_files if '_NS.csv' in f]
print(f"\n📊 Available stocks: {stock_names}")

# Step 2: Function to load and preprocess data from CSV
def load_stock_data(csv_filename):
    """
    Load stock data from CSV file and preprocess it.
    
    Args:
        csv_filename: Name of the CSV file (e.g., 'TCS_NS.csv')
    
    Returns:
        DataFrame with preprocessed stock data
    """
    csv_path = os.path.join(data_folder, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
    
    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    # Sort by date
    df.sort_index(inplace=True)
    
    # Drop any missing values
    df = df.dropna()
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Missing required columns in {csv_filename}")
        return None
    
    print(f"✅ Loaded {csv_filename}: {len(df)} rows, Date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df

# Test loading one file
if csv_files:
    test_file = csv_files[0]
    test_data = load_stock_data(test_file)
    if test_data is not None:
        print(f"\n📊 Sample data from {test_file}:")
        print(test_data.head())

# Step 3: Function to train model for a single stock
def train_stock_model(stock_name):
    """
    Train a Linear Regression model for a stock using CSV data.
    
    Args:
        stock_name: Stock name (e.g., 'TCS', 'RELIANCE')
    
    Returns:
        Dictionary with model metrics or None if training fails
    """
    csv_filename = f"{stock_name}_NS.csv"
    csv_path = os.path.join(data_folder, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️ CSV file not found: {csv_filename}")
        return None
    
    # Load data
    df = load_stock_data(csv_filename)
    if df is None or len(df) < 50:
        print(f"  ⚠️ Not enough data for {stock_name} (need at least 50 rows)")
        return None
    
    # Prepare features and target
    # Using Open, High, Low, Volume to predict Close
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    
    # Split data (80% train, 20% test, no shuffle to preserve time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Save model
    model_filename = f"{stock_name}_NS_model.joblib"
    model_path = os.path.join(models_folder, model_filename)
    dump(model, model_path)
    
    print(f"  ✅ Model saved: {model_filename}")
    print(f"     Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
    print(f"     Test  MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
    
    return {
        'Stock': stock_name,
        'Train_MAE': round(train_mae, 3),
        'Train_RMSE': round(train_rmse, 3),
        'Test_MAE': round(test_mae, 3),
        'Test_RMSE': round(test_rmse, 3),
        'Test_R2': round(test_r2, 4),
        'Data_Points': len(df)
    }

# Step 4: Train models for all available stocks
print("\n" + "=" * 70)
print("🚀 TRAINING MODELS FOR ALL STOCKS")
print("=" * 70)

results = []
for stock_name in stock_names:
    print(f"\n📈 Training model for {stock_name}...")
    result = train_stock_model(stock_name)
    if result:
        results.append(result)

# Create accuracy report
if results:
    report_df = pd.DataFrame(results)
    report_df.to_csv('accuracy_report.csv', index=False)
    print("\n" + "=" * 70)
    print("📊 ACCURACY REPORT")
    print("=" * 70)
    print(report_df.to_string(index=False))
    print(f"\n✅ Accuracy report saved to: accuracy_report.csv")
else:
    print("\n❌ No models were trained successfully.")

# Step 7: Batch predictions for all stocks
print("\n" + "=" * 70)
print("🔮 BATCH PREDICTIONS FOR ALL STOCKS")
print("=" * 70)

predictions_summary = []
for stock_name in stock_names:
    csv_filename = f"{stock_name}_NS.csv"
    model_filename = f"{stock_name}_NS_model.joblib"
    
    csv_path = os.path.join(data_folder, csv_filename)
    model_path = os.path.join(models_folder, model_filename)
    
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        continue
    
    try:
        # Load data and model
        df = load_stock_data(csv_filename)
        if df is None or len(df) == 0:
            continue
        
        model = load(model_path)
        
        # Get latest data
        latest_data = df.tail(1)
        X_latest = latest_data[['Open', 'High', 'Low', 'Volume']]
        
        # Predict
        next_day_pred = model.predict(X_latest)[0]
        last_close = latest_data['Close'].iloc[0]
        last_date = latest_data.index[0]
        
        change = next_day_pred - last_close
        change_pct = (change / last_close) * 100
        
        predictions_summary.append({
            'Stock': stock_name,
            'Last_Date': last_date.date(),
            'Last_Close': round(last_close, 2),
            'Predicted_Close': round(next_day_pred, 2),
            'Change': round(change, 2),
            'Change_%': round(change_pct, 2),
            'Signal': 'BUY' if change > 0 else 'SELL'
        })
        
        print(f"✅ {stock_name}: ₹{last_close:.2f} → ₹{next_day_pred:.2f} ({change_pct:+.2f}%)")
        
    except Exception as e:
        print(f"❌ Error predicting for {stock_name}: {str(e)}")
        continue

if predictions_summary:
    summary_df = pd.DataFrame(predictions_summary)
    print("\n" + "=" * 70)
    print("📊 PREDICTIONS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("\n✅ All predictions completed!")
    print("\n🎉 Model training and prediction completed successfully!")
else:
    print("\n❌ No predictions were made.")

