import os
import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.svm import SVR  # 使用回歸 SVM
from xgboost import XGBRegressor  # 使用回歸 XGBoost
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a dataframe.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    for column in df.columns:
        if column == 'hospital_death':  # Don't impute target variable
            continue
            
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            if df[column].dtype in ['float64', 'int64']:
                # For numeric columns, replace NaN with median (more robust than mean)
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                logging.debug(f"Filled {missing_count} missing values in {column} with median {median_value}")
            elif df[column].dtype == 'object':
                # For categorical columns, replace NaN with mode
                mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                df[column].fillna(mode_value, inplace=True)
                logging.debug(f"Filled {missing_count} missing values in {column} with mode {mode_value}")
    
    return df


def pipeline(
    df_or_path: Union[str, pd.DataFrame], 
    encoders: Optional[Dict] = None, 
    selected_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    Process dataframe or load from path, handle missing values and encode categorical features.
    
    Args:
        df_or_path: DataFrame or path to CSV file
        encoders: Dictionary of category encoders for consistent transforms
        selected_features: List of selected features to include
        
    Returns:
        Tuple containing:
        - Processed DataFrame
        - Encounter IDs (if available)
        - Encoders dictionary
    """
    # Handle input that could be either a DataFrame or a path
    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            raise FileNotFoundError(f"File not found: {df_or_path}")
        df = pd.read_csv(df_or_path)
        logging.info(f"Loaded data from {df_or_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        df = df_or_path.copy()
        logging.info(f"Processing provided DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Save encounter_id for submission if it exists
    encounter_ids = None
    if 'encounter_id' in df.columns:
        encounter_ids = df['encounter_id'].copy()
    
    # Drop identifier columns if they exist
    id_columns = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
    df = df.drop(columns=[col for col in id_columns if col in df.columns])

    # Handle missing values
    df = handle_missing_values(df)

    # Define the categorical columns
    cat_columns = [col for col in df.columns if df[col].dtype == 'object']
    logging.info(f"Found {len(cat_columns)} categorical columns")
    
    # Initialize encoders dictionary if not provided
    if encoders is None:
        encoders = {}
        
    # Encode categorical features
    for column in cat_columns:
        if column not in encoders:
            # Create new mapping for training data
            unique_values = df[column].unique()
            encoders[column] = {value: index for index, value in enumerate(unique_values)}
        
        # Apply mapping using the dictionary
        df[column] = df[column].map(encoders[column])
        
        # Handle unseen categories in test data
        df[column].fillna(-1, inplace=True)  # Use -1 for unseen categories
        df[column] = df[column].astype(int)  # Convert to integers
    
    # Filter to selected features if provided
    if selected_features is not None:
        # Add missing columns with zeros
        for feature in selected_features:
            if feature not in df.columns and feature != 'hospital_death':
                df[feature] = 0
                
        # Only keep selected features (and target if present)
        keep_columns = selected_features.copy()
        if 'hospital_death' in df.columns and 'hospital_death' not in keep_columns:
            keep_columns.append('hospital_death')
            
        df = df[keep_columns]
        logging.info(f"Filtered to {len(keep_columns)} selected features")
    
    return df, encounter_ids, encoders
np.random.seed(42)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
    
logging.info("Loading and processing training data...")
train_path = "datasets/train.csv"
    
try:
    train_df, train_encounter_ids, encoders = pipeline(train_path)
except FileNotFoundError as e:
    logging.error(f"Error: {e}")
    logging.error("Please ensure the dataset file exists at the specified path.")


if 'hospital_death' not in train_df.columns:
        logging.error("Error: 'hospital_death' column not found in the data.")


    # Save encoders for future use


X = train_df.drop('hospital_death', axis=1)
y = train_df['hospital_death']

    # Check for class imbalance
class_counts = y.value_counts()
logging.info(f"Class distribution: {class_counts.to_dict()}")
imbalance_ratio = class_counts[0] / class_counts[1]
logging.info(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
    
    # Handle remaining categorical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_features:
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset indices for consistency
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# 1. Extract data with hospital_death = 1 (minority class)
X_class1 = X_train[y_train == 1]
y_class1 = y_train[y_train == 1]

print(f"Original class distribution: Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")

# 2. Define number of synthetic samples to generate
num_samples = 2500

# 3. Create an empty DataFrame with the same columns as X_train to store synthetic samples
synthetic_data = pd.DataFrame(columns=X_train.columns)

# 4. Sample from minority class to create base samples (with replacement)
row_indices = np.random.choice(len(X_class1), num_samples, replace=True)
synthetic_data = X_class1.iloc[row_indices].copy().reset_index(drop=True)

# 5. Train models and generate synthetic features
for column in X_train.columns:
    # Skip columns with only a single unique value
    if X_class1[column].nunique() <= 1:
        continue
    
    # Prepare inputs (all other columns) and target (current column)
    X_inputs = X_class1.drop(columns=[column])
    y_target = X_class1[column]
    
    try:
        # Train SVM regressor for this feature
        svm = SVR()
        svm_param_grid = {
    "C": np.logspace(-5, 5, 20),
    "gamma": np.logspace(-5, 5, 20),
    "epsilon": np.linspace(0.0001, 1.0, 50),
    "kernel": ["linear", "rbf", "poly", "sigmoid"]
}
        svm_search = RandomizedSearchCV(svm, svm_param_grid, n_iter=2500, cv=3, 
                                       n_jobs=-1, random_state=42)
        svm_search.fit(X_inputs, y_target)
        
        # Train XGBoost regressor for this feature
        xgb = XGBRegressor()
        xgb_param_grid = {
    "n_estimators": np.arange(50, 500, 50),
    "learning_rate": np.logspace(-4, 0, 20),
    "max_depth": np.arange(2, 15),
    "min_child_weight": np.logspace(-2, 2, 20),
    "subsample": np.linspace(0.1, 1.0, 20),
    "colsample_bytree": np.linspace(0.1, 1.0, 20)
}
        xgb_search = RandomizedSearchCV(xgb, xgb_param_grid, n_iter=2500, cv=3, 
                                       n_jobs=-1, random_state=42)
        xgb_search.fit(X_inputs, y_target)
        
        # Use trained models to predict the column values for the synthetic samples
        synthetic_inputs = synthetic_data.drop(columns=[column])
        svm_preds = svm_search.predict(synthetic_inputs)
        xgb_preds = xgb_search.predict(synthetic_inputs)
        
        # Average the predictions and add small random noise
        avg_preds = (svm_preds + xgb_preds) / 2
        col_std = X_class1[column].std() * 0.1  # 10% of standard deviation for noise
        avg_preds += np.random.normal(0, col_std, num_samples)
        
        # Update the synthetic data with the new values
        synthetic_data[column] = avg_preds
        
        # Log the progress
        logging.info(f"Generated synthetic values for column: {column}")
        
    except Exception as e:
        logging.warning(f"Error generating synthetic values for {column}: {e}")
        # Keep the original sampled values if there's an error

# 6. Create synthetic labels (all 1s for minority class)
synthetic_labels = np.ones(num_samples)

# 7. Combine original data with synthetic data
X_train_augmented = pd.concat([X_train, synthetic_data], ignore_index=True)
y_train_augmented = np.concatenate([y_train, synthetic_labels])

print(f"Augmented class distribution: Class 0: {sum(y_train_augmented == 0)}, Class 1: {sum(y_train_augmented == 1)}")

# 8. Save augmented dataset
df_augmented = X_train_augmented.copy()
df_augmented['hospital_death'] = y_train_augmented

output_path = "models/train_augmented.csv"
df_augmented.to_csv(output_path, index=False)

print(f"✅ Augmented data saved to {output_path}")
print(f"🔹 Original X_train shape: {X_train.shape}")
print(f"🔹 Augmented X_train shape: {X_train_augmented.shape}")