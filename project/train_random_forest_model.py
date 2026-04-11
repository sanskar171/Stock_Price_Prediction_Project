"""
Train Random Forest Models for Stock Prediction
This is the second model type (first was Linear Regression)
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🌲 TRAINING RANDOM FOREST MODELS (Second Model Type)")
print("=" * 80)

data_folder = "data"
models_folder = "models"

# Create models folder if it doesn't exist
os.makedirs(models_folder, exist_ok=True)

# Discover available CSV files
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
stock_names = [f.replace('_NS.csv', '') for f in csv_files if '_NS.csv' in f]

print(f"\n📁 Found {len(stock_names)} stocks to train:")
for stock in stock_names:
    print(f"   - {stock}")

def load_stock_data(csv_filename):
    """Load and preprocess stock data"""
    csv_path = os.path.join(data_folder, csv_filename)
    
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None
    
    return df

def train_random_forest_model(stock_name):
    """Train Random Forest model for a stock"""
    csv_filename = f"{stock_name}_NS.csv"
    csv_path = os.path.join(data_folder, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️ CSV file not found: {csv_filename}")
        return None
    
    # Load data
    df = load_stock_data(csv_filename)
    if df is None or len(df) < 50:
        print(f"  ⚠️ Not enough data for {stock_name}")
        return None
    
    # Prepare features and target
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train Random Forest model
    # Using optimized hyperparameters to prevent overfitting
    model = RandomForestRegressor(
        n_estimators=50,       # Fewer trees to prevent overfitting
        max_depth=5,           # Shallower trees
        min_samples_split=20,  # More samples needed to split
        min_samples_leaf=10,   # More samples in leaf nodes
        max_features='sqrt',   # Use sqrt of features for each tree
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    print(f"\n📈 Training Random Forest for {stock_name}...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Save model with RF suffix to distinguish from Linear Regression
    model_filename = f"{stock_name}_NS_RF_model.joblib"
    model_path = os.path.join(models_folder, model_filename)
    dump(model, model_path)
    
    print(f"  ✅ Model saved: {model_filename}")
    print(f"     Train MAE: {train_mae:.4f} ₹, RMSE: {train_rmse:.4f} ₹, R²: {train_r2:.6f}")
    print(f"     Test  MAE: {test_mae:.4f} ₹, RMSE: {test_rmse:.4f} ₹, R²: {test_r2:.6f}")
    
    return {
        'Stock': stock_name,
        'Model_Type': 'Random Forest',
        'Train_MAE': round(train_mae, 3),
        'Train_RMSE': round(train_rmse, 3),
        'Train_R2': round(train_r2, 4),
        'Test_MAE': round(test_mae, 3),
        'Test_RMSE': round(test_rmse, 3),
        'Test_R2': round(test_r2, 4),
        'Data_Points': len(df)
    }

# Train models for all stocks
print("\n" + "=" * 80)
print("🚀 TRAINING RANDOM FOREST MODELS FOR ALL STOCKS")
print("=" * 80)

results = []
for stock_name in stock_names:
    result = train_random_forest_model(stock_name)
    if result:
        results.append(result)

# Create accuracy report for Random Forest
if results:
    rf_report_df = pd.DataFrame(results)
    rf_report_df.to_csv('accuracy_report_RF.csv', index=False)
    
    print("\n" + "=" * 80)
    print("📊 RANDOM FOREST MODEL PERFORMANCE")
    print("=" * 80)
    print(rf_report_df.to_string(index=False))
    print(f"\n✅ Random Forest accuracy report saved to: accuracy_report_RF.csv")
    
    # Compare with Linear Regression
    if os.path.exists('accuracy_report.csv'):
        lr_report = pd.read_csv('accuracy_report.csv')
        
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON: Linear Regression vs Random Forest")
        print("=" * 80)
        
        comparison_data = []
        for stock in stock_names:
            lr_row = lr_report[lr_report['Stock'] == stock]
            rf_row = rf_report_df[rf_report_df['Stock'] == stock]
            
            if not lr_row.empty and not rf_row.empty:
                comparison_data.append({
                    'Stock': stock,
                    'LR_R2': lr_row.iloc[0]['Test_R2'],
                    'RF_R2': rf_row.iloc[0]['Test_R2'],
                    'LR_MAE': lr_row.iloc[0]['Test_MAE'],
                    'RF_MAE': rf_row.iloc[0]['Test_MAE'],
                    'Better_Model': 'Random Forest' if rf_row.iloc[0]['Test_R2'] > lr_row.iloc[0]['Test_R2'] else 'Linear Regression'
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\nComparison by R² Score:")
            print("-" * 80)
            print(f"{'Stock':<12} | LR R²      | RF R²      | Better Model")
            print("-" * 80)
            for _, row in comparison_df.iterrows():
                print(f"{row['Stock']:<12} | {row['LR_R2']:.6f} | {row['RF_R2']:.6f} | {row['Better_Model']}")
            
            print("\n" + "-" * 80)
            print(f"Average LR R²: {comparison_df['LR_R2'].mean():.6f}")
            print(f"Average RF R²: {comparison_df['RF_R2'].mean():.6f}")
            
            better_rf = (comparison_df['Better_Model'] == 'Random Forest').sum()
            better_lr = (comparison_df['Better_Model'] == 'Linear Regression').sum()
            print(f"\nRandom Forest better: {better_rf} stocks")
            print(f"Linear Regression better: {better_lr} stocks")
            
            comparison_df.to_csv('model_comparison.csv', index=False)
            print(f"\n✅ Model comparison saved to: model_comparison.csv")
else:
    print("\n❌ No models were trained successfully.")

print("\n" + "=" * 80)
print("✅ RANDOM FOREST TRAINING COMPLETE")
print("=" * 80)
print("""
📁 Files Generated:
   • accuracy_report_RF.csv - Random Forest performance metrics
   • model_comparison.csv - Comparison between Linear Regression and Random Forest
   • models/*_RF_model.joblib - Trained Random Forest model files

🎯 You now have TWO model types trained:
   1. Linear Regression (models/*_model.joblib)
   2. Random Forest (models/*_RF_model.joblib)
""")

