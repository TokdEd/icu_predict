import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import csv
import warnings
from sklearn.metrics import roc_curve
import os
import logging
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
def pipeline(df_or_path):
    """
    Process dataframe or load from path, handle missing values and encode categorical features.
    
    Args:
        df_or_path: DataFrame or path to CSV file
        
    Returns:
        Processed DataFrame
    """
    # Handle input that could be either a DataFrame or a path
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()
    
    # Drop identifier columns
    to_drop = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
    drop_cols = [col for col in to_drop if col in df.columns]
    # Save encounter_id for submission if it exists
    encounter_ids = None
    if 'encounter_id' in df.columns:
        encounter_ids = df['encounter_id'].copy()
    
    df = df.drop(columns=drop_cols)

    # Handle missing values
    for column in df.columns:
        if df[column].dtype == 'float64':
            # Handle numeric columns (replace NaN with mean)
            df[column].fillna(df[column].mean(), inplace=True)
        elif df[column].dtype == 'object':
            # Handle categorical columns (replace NaN with mode)
            df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else "Unknown", inplace=True)

    # Create a dictionary to store the mappings
    columns_to_map = {}

    # Define the columns to process
    cat_columns = ["ethnicity", "gender", "icu_admit_source", "icu_stay_type", 
                 "icu_type", "apache_3j_bodysystem", "apache_2_bodysystem"]
    
    # Only process columns that actually exist in the dataframe
    columns = [col for col in cat_columns if col in df.columns]

    # Iterate through the columns and create mappings
    for column in columns:
        unique_values = df[column].unique()
        mapping = {value: index for index, value in enumerate(unique_values)}
        columns_to_map[column] = mapping
        # Apply mapping
        df[column] = df[column].map(mapping)

    return df, encounter_ids

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """
    Calculate Mean Absolute Percentage Error with handling for zero values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = (y_true != 0) & (y_pred != 0)
    absolute_percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / (y_true[non_zero_mask] + epsilon))
    return np.mean(absolute_percentage_error) * 100 if len(absolute_percentage_error) > 0 else np.nan

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
'''def greedy_feature_selection(data, target, features, max_features=85, objective='binary:logistic'):
    selected_features = []
    score_history = []
    current_best_score = 0
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold cross-validation
    
    for _ in range(min(max_features, len(features))):
        best_feature = None
        best_score = current_best_score
        
        for feature in features:
            if feature in selected_features:
                continue
                
            trial_features = selected_features + [feature]
            X_trial = data[trial_features]
            model = xgb.XGBClassifier(objective=objective, eval_metric='auc', random_state=42)
            
            # Use cross-validation to calculate AUC
            trial_score = np.mean(cross_val_score(model, X_trial, target, cv=cv, scoring='roc_auc'))
            
            if trial_score > best_score:
                best_score = trial_score
                best_feature = feature
        
        if best_feature:
            selected_features.append(best_feature)
            score_history.append(best_score)
            current_best_score = best_score
            print(f"Selected feature: {best_feature}, Cross-validated ROC AUC: {current_best_score:.4f}")
        
    return selected_features, score_history'''


def save_submission(encounter_ids, predictions, filename):
    """
    Save predictions to a CSV file for submission.
    
    Args:
        encounter_ids: Series of encounter IDs
        predictions: Array of binary predictions
        filename: Output filename
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['encounter_id', 'hospital_death'])
        for i in range(len(predictions)):
            writer.writerow([encounter_ids.iloc[i], int(predictions[i])])
    print(f"Submission saved to {filename}")

if __name__ == "__main__":
    # Read and process data
    print("Loading and processing training data...")
    train_path = "models/train_augmented.csv"
    train_df, train_encounter_ids = pipeline(train_path)
    
    # Check if target column exists
    if 'hospital_death' not in train_df.columns:
        print("Error: 'hospital_death' column not found in the data. Please check your dataset.")
        exit(1)
    
    # Split features and target
    X = train_df.drop('hospital_death', axis=1)
    y = train_df['hospital_death']
    
    # Handle categorical features if any exist
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Train-test split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    train_encounter_ids_split = None
    if train_encounter_ids is not None:
        train_encounter_ids_split = train_encounter_ids[split_index:]
    
    # Drop any columns with NaN values or handle them appropriately
    columns_with_nan = X_train.columns[X_train.isna().any()].tolist()
    if columns_with_nan:
        print(f"Dropping columns with NaN values: {columns_with_nan}")
        X_train = X_train.drop(columns=columns_with_nan)
        X_test = X_test.drop(columns=columns_with_nan)
    
    # Check if pre-selected features exist
    if os.path.exists("models/selected_features.txt"):
        logging.info("Loading pre-selected features...")
        with open("models/selected_features.txt", "r") as f:
            selected_features = [line.strip() for line in f]
            '''else:
        # Use greedy feature selection
        print("Starting feature selection...")
        selected_features, score_history = greedy_feature_selection(
            X_train, 
            y_train, 
            features=X_train.columns.tolist(),
            max_features=50,  # Reduced to improve speed
            objective='binary:logistic'  # For binary classification
        )
        
        # Save selected features
        os.makedirs("models", exist_ok=True)
        with open("models/selected_features.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")'''
    
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Random search for hyperparameter tuning
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    param_dist = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],  
    'learning_rate': [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],  
    'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
    'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  
    'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]  
}
    
    # Use RandomizedSearchCV instead of GridSearchCV
    random_search = RandomizedSearchCV(
        xgb.XGBClassifier(objective='binary:logistic', random_state=42, 
                          eval_metric='logloss'),
        param_distributions=param_dist,
        n_iter=1000,  # Number of parameter settings sampled
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    #{'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.1, 'colsample_bytree': 1.0}
    
    random_search.fit(X_train_selected, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    # Make predictions on validation set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Convert probabilities to binary predictions using optimal threshold
    y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)
    
    # Calculate metrics for validation set
    roc = roc_auc_score(y_test, y_pred_proba)
    mse = mean_squared_error(y_test, y_pred_binary)
    mae = mean_absolute_error(y_test, y_pred_binary)
    
    print(f"Validation Mean Squared Error (MSE): {mse:.4f}")
    print(f"Validation Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Validation ROC AUC Score: {roc:.4f}")
    
    # Save validation predictions if encounter IDs are available
    if train_encounter_ids_split is not None:
        save_submission(train_encounter_ids_split, y_pred_binary, 'validation_predictions.csv')
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    best_model.save_model("models/best_xgboost_model.json")
    print("Best model saved to models/best_xgboost_model.json")
    
    # Process test data
    print("Loading and processing test data...")
    test_path = "datasets/test.csv"
    test_df, test_encounter_ids = pipeline(test_path)
    
    # Ensure test data has the same features as training data
    for feature in selected_features:
        if feature not in test_df.columns:
            print(f"Feature {feature} not found in test data. Adding with zeros.")
            test_df[feature] = 0
    
    X_test_submission = test_df[selected_features]
    params = {'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0.1, 'colsample_bytree': 1.0}
    # Make predictions on test data using only the best XGBoost model
    y_pred_proba_test = best_model.predict_proba(X_test_submission)[:, 1]
    print("Data type:", y_pred_proba_test.dtype) 
    # Convert to binary predictions using the optimal threshold
    y_pred_binary_test = (y_pred_proba_test > optimal_threshold).astype(int)
    
    # Save test predictions
    if test_encounter_ids is not None:
        save_submission(test_encounter_ids, y_pred_binary_test, 'submission_xg.csv')
    else:
        print("Warning: No encounter IDs found for test data. Cannot create submission file.")
    
    # Feature importance visualization
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(best_model, max_num_features=20)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    
    # Plot ROC AUC improvement with feature selection
    '''if 'score_history' in locals():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(score_history) + 1), score_history, marker='o', linestyle='-')
        plt.xlabel("Number of Features Selected")
        plt.ylabel("ROC AUC Score")
        plt.title("Performance Improvement with Feature Selection (Greedy Strategy)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("feature_selection_performance.png", dpi=300)'''
    
    
    # Plot ROC curve for validation set
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("roc_curve.png", dpi=300)
    
    print("Analysis complete. Results and visualizations saved.")