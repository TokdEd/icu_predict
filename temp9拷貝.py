import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve
import xgboost as xgb
from xgboost import callback
import matplotlib.pyplot as plt
import csv
import os
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the selected features based on greedy feature selection
SELECTED_FEATURES = [
    'apache_4a_hospital_death_prob',
    'd1_mbp_min',
    'h1_resprate_min',
    'ventilated_apache',
    'd1_temp_max',
    'd1_sysbp_noninvasive_max',
    'd1_potassium_min'
]
def pipeline＿test(df_or_path, keep_only_selected=True):
    """
    Process dataframe or load from path, handle missing values and keep only selected features.
    
    Args:
        df_or_path: DataFrame or path to CSV file
        keep_only_selected: If True, keep only the selected features
        
    Returns:
        Processed DataFrame and encounter_ids
    """
    # Handle input that could be either a DataFrame or a path
    if isinstance(df_or_path, str):
        print(f"Loading data from {df_or_path}...")
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()
    
    # Save encounter_id for submission if it exists
    encounter_ids = None
    if 'encounter_id' in df.columns:
        encounter_ids = df['encounter_id'].copy()
    
    # Drop identifier columns
    id_columns = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
    drop_cols = [col for col in id_columns if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    if keep_only_selected:
        # Keep only selected features and target if it exists
        columns_to_keep = SELECTED_FEATURES.copy()
        if 'hospital_death' in df.columns:
            columns_to_keep.append('hospital_death')
        
        # Check which selected features exist in the dataframe
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        
        # Only keep existing selected features
        df = df[existing_columns]
        
        # Check for missing selected features
        missing_features = set(SELECTED_FEATURES) - set(df.columns)
        if missing_features:
            print(f"Warning: The following selected features are missing from the dataset: {missing_features}")
        # Handle missing values for each column
    for column in df.columns:
        if column != 'hospital_death':  # Skip target variable
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                # Replace NaN with mean for numeric columns
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                print(f"Filled missing values in {column} with mean: {mean_value:.2f}")
            elif df[column].dtype == 'object':
                # Replace NaN with mode for categorical columns
                mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                df[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {column} with mode: {mode_value}")
    
    return df, encounter_ids
def pipeline(df_or_path, keep_only_selected=True):
    """
    Process dataframe or load from path, handle missing values and keep only selected features.
    
    Args:
        df_or_path: DataFrame or path to CSV file
        keep_only_selected: If True, keep only the selected features
        
    Returns:
        Processed DataFrame and encounter_ids
    """
    # Handle input that could be either a DataFrame or a path
    if isinstance(df_or_path, str):
        print(f"Loading data from {df_or_path}...")
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()
    
    # Save encounter_id for submission if it exists
    encounter_ids = None
    if 'encounter_id' in df.columns:
        encounter_ids = df['encounter_id'].copy()
    
    # Drop identifier columns
    id_columns = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
    drop_cols = [col for col in id_columns if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    if keep_only_selected:
        # Keep only selected features and target if it exists
        columns_to_keep = SELECTED_FEATURES.copy()
        if 'hospital_death' in df.columns:
            columns_to_keep.append('hospital_death')
        
        # Check which selected features exist in the dataframe
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        
        # Only keep existing selected features
        df = df[existing_columns]
        
        # Check for missing selected features
        missing_features = set(SELECTED_FEATURES) - set(df.columns)
        if missing_features:
            print(f"Warning: The following selected features are missing from the dataset: {missing_features}")
    if 'apache_4a_hospital_death_prob' in df.columns:
        original_size = len(df)
        df = df[df['apache_4a_hospital_death_prob'] >= 0]  # 保留值 >= 0 的數據
        removed_rows = original_size - len(df)
        print(f"Removed {removed_rows} rows with invalid apache_4a_hospital_death_prob (< 0)")
    # Handle missing values for each column
    for column in df.columns:
        if column != 'hospital_death':  # Skip target variable
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                # Replace NaN with mean for numeric columns
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                print(f"Filled missing values in {column} with mean: {mean_value:.2f}")
            elif df[column].dtype == 'object':
                # Replace NaN with mode for categorical columns
                mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                df[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {column} with mode: {mode_value}")
    
    return df, encounter_ids

def apply_gaussian_noise(data, target_column, positive_class_value=1, noise_level=0.05, multiplier=2):
    """
    Apply Gaussian noise to generate synthetic samples for the minority class.
    
    Args:
        data: DataFrame containing the data
        target_column: Name of the target column
        positive_class_value: Value representing the positive class
        noise_level: Standard deviation of the Gaussian noise
        multiplier: How many times to multiply the minority class
        
    Returns:
        DataFrame with augmented data
    """
    # Separate majority and minority classes
    majority_class = data[data[target_column] != positive_class_value]
    minority_class = data[data[target_column] == positive_class_value]
    
    print(f"Original class distribution:")
    print(f"  Majority class (0): {len(majority_class)} samples")
    print(f"  Minority class (1): {len(minority_class)} samples")
    
    # Create synthetic samples with Gaussian noise
    synthetic_samples = []
    
    # For continuous features, use Gaussian noise
    # For binary features, we'll handle differently
    continuous_features = [col for col in minority_class.columns if col != target_column and col != 'ventilated_apache']
    binary_features = ['ventilated_apache']
    
    for _ in range(multiplier - 1):
        for _, row in minority_class.iterrows():
            new_sample = row.copy()
            
            # Add Gaussian noise to continuous features
            for feature in continuous_features:
                feature_std = data[feature].std()
                noise = np.random.normal(0, noise_level * feature_std)
                new_sample[feature] += noise
            
            # For binary features, flip with small probability (10%)
            for feature in binary_features:
                if np.random.random() < 0.1:  # 10% chance to flip
                    new_sample[feature] = 1 - new_sample[feature]
            
            synthetic_samples.append(new_sample)
    
    # Convert list of samples to DataFrame
    synthetic_df = pd.DataFrame(synthetic_samples)
    
    # Combine original data with synthetic samples
    augmented_data = pd.concat([data, synthetic_df], ignore_index=True)
    
    print(f"Augmented class distribution:")
    print(f"  Majority class (0): {len(majority_class)} samples")
    print(f"  Minority class (1): {len(minority_class) * multiplier} samples")
    
    return augmented_data
def apply_smote_augmentation(data, target_column, multiplier=2, random_state=42, k_neighbors=5):
    """
    Apply SMOTE to generate synthetic samples for the minority class with rounding and duplicate prevention.
    
    Args:
        data: DataFrame containing the data.
        target_column: Name of the target column.
        multiplier: How many times to multiply the minority class.
        random_state: Random seed for reproducibility.
        k_neighbors: Number of nearest neighbors to use for SMOTE.
    
    Returns:
        DataFrame with augmented data.
    """
    from imblearn.over_sampling import SMOTE
    import pandas as pd
    import numpy as np
    
    # Identify numerical features (exclude target and binary features)
    binary_features = ['ventilated_apache']  # Binary variables to exclude
    features = [col for col in data.columns if col != target_column and col not in binary_features]
    
    # Round original data to 3 decimal places
    data_rounded = data.copy()
    for col in features:
        data_rounded[col] = data_rounded[col].round(3)
    
    # Get X (features) and y (labels)
    X = data_rounded[features]
    y = data_rounded[target_column]
    
    # Calculate target class counts
    majority_class_count = (y == 0).sum()
    minority_class_count = (y == 1).sum()
    target_minority_count = int(minority_class_count * multiplier)  # Target minority class count
    
    print(f"Original class distribution:")
    print(f" Majority class (0): {majority_class_count} samples")
    print(f" Minority class (1): {minority_class_count} samples")
    
    # Handle case where minority class is too small for default k_neighbors
    if minority_class_count <= k_neighbors:
        adjusted_k = max(1, minority_class_count - 1)
        print(f"Warning: Adjusting k_neighbors from {k_neighbors} to {adjusted_k} due to small minority class size")
        k_neighbors = adjusted_k
    
    # Apply SMOTE with adjusted parameters
    smote = SMOTE(
        sampling_strategy={1: target_minority_count}, 
        random_state=random_state,
        k_neighbors=k_neighbors
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create new DataFrame for augmented data
    augmented_data = pd.DataFrame(X_resampled, columns=features)
    
    # Round synthetic samples to 3 decimal places
    for col in features:
        augmented_data[col] = augmented_data[col].round(3)
    
    augmented_data[target_column] = y_resampled
    
    # Handle binary features
    if binary_features:
        # Extract original data instances
        original_indices = range(len(data))
        synthetic_indices = range(len(data), len(augmented_data))
        
        # For original data points, keep original binary values
        for feature in binary_features:
            augmented_data.loc[original_indices, feature] = data[feature].values
        
        # For synthetic samples, assign binary values based on nearest neighbors
        minority_indices = data[data[target_column] == 1].index
        
        for i in synthetic_indices:
            # Find the original instances used to create this synthetic sample
            synthetic_features = augmented_data.loc[i, features].values.reshape(1, -1)
            
            # Find nearest neighbors in minority class
            distances = np.linalg.norm(
                X.loc[minority_indices].values - synthetic_features, 
                axis=1
            )
            nearest_idx = minority_indices[np.argmin(distances)]
            
            # Assign binary features from nearest neighbor
            for feature in binary_features:
                augmented_data.loc[i, feature] = data.loc[nearest_idx, feature]
    
    # Add small differentiation to reduce exact duplicates
    def add_tiny_variation(group):
        # If duplicates exist in the group, add tiny variations
        if len(group) > 1:
            # Create a small random variation
            variation = np.random.normal(0, 0.001, size=len(group))
            
            # Apply variation to all columns except binary features and target
            for col in features:
                group[col] += variation
                # Re-round to 3 decimal places
                group[col] = group[col].round(3)
        return group
    
    # Group by features to identify and slightly modify duplicates
    augmented_data = augmented_data.groupby(features + binary_features + [target_column], group_keys=False).apply(add_tiny_variation)
    
    # Final rounding and duplicate check
    for col in features:
        augmented_data[col] = augmented_data[col].round(3)
    
    print(f"Augmented class distribution:")
    print(f" Majority class (0): {(augmented_data[target_column] == 0).sum()} samples")
    print(f" Minority class (1): {(augmented_data[target_column] == 1).sum()} samples")
    
    # Check for duplicates
    duplicate_count = augmented_data.duplicated().sum()
    if duplicate_count > 0:
        print(f"Warning: {duplicate_count} duplicate rows found in augmented data")
        # Optional: remove duplicates if needed
        augmented_data = augmented_data.drop_duplicates()
    
    return augmented_data
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

def main():
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    
    # Read and process data
    print("===== Loading and processing training data =====")
    train_path = "datasets/train.csv"
    train_df, train_encounter_ids = pipeline(train_path)
    
    # Check if target column exists
    if 'hospital_death' not in train_df.columns:
        print("Error: 'hospital_death' column not found in the data. Please check your dataset.")
        exit(1)
    train_df[SELECTED_FEATURES].to_csv("datasets/train_selected.csv", index=False)
    # Save processed dataframe
    processed_train_path = "datasets/train_processed.csv"
    train_df.to_csv(processed_train_path, index=False)
    print(f"Processed training data saved to {processed_train_path}")
    
    # Apply data augmentation to balance the dataset
    print("\n===== Applying data augmentation =====")
    '''augmented_df = apply_gaussian_noise(
        train_df, 
        target_column='hospital_death', 
        positive_class_value=1, 
        noise_level=0.05,
        multiplier=2  # Double the minority class
    )'''
    
    augmented_df = apply_smote_augmentation(train_df, target_column="hospital_death", multiplier=2)
    
    # Save augmented dataframe
    augmented_train_path = "datasets/train_augmented.csv"
    augmented_df.to_csv(augmented_train_path, index=False)
    print(f"Augmented training data saved to {augmented_train_path}")
    
    # Split features and target
    X = augmented_df.drop('hospital_death', axis=1)
    y = augmented_df['hospital_death']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Print feature statistics for reference
    print("\n===== Feature Statistics =====")
    for feature in SELECTED_FEATURES:
        if feature in X.columns:
            print(f"{feature}:")
            print(f"  Mean: {X[feature].mean():.4f}")
            print(f"  Std: {X[feature].std():.4f}")
            print(f"  Min: {X[feature].min():.4f}")
            print(f"  Max: {X[feature].max():.4f}")
    
    # Train XGBoost model with optimized hyperparameters
    print("\n===== Training XGBoost model =====")
    param_dist = {
        'subsample': np.linspace(0.5, 1.0, 6),
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_weight': [1, 2, 3, 4, 5],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'gamma': np.linspace(0, 0.5, 6),
        'colsample_bytree': np.linspace(0.5, 1.0, 6),
        'objective': ['binary:logistic'],
        'eval_metric': ['auc'],
        'random_state': [42]
    }

    # Initialize XGBClassifier
    model = xgb.XGBClassifier(random_state=42)

    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=3000,  # Number of random combinations to try
        scoring='roc_auc',  # We want to maximize AUC
        cv=3,  # 3-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model to the training data
    random_search.fit(X_train, y_train)

    # Output the best parameters and best AUC score
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best AUC score: {random_search.best_score_}")

    # Get the best model from the random search
    model = random_search.best_estimator_

    # Create evaluation set for monitoring
    eval_set = [(X_test, y_test)]

    # Train the model using the best found hyperparameters
    model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        verbose=True
    )

    
    print("\n===== Model Evaluation =====")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    
    print(f"Test Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test ROC AUC Score: {roc:.4f}")
    
    # Save the best model
    model.save_model("models/best_xgboost_model.json")
    print("Best model saved to models/best_xgboost_model.json")
    
    # Process test data
    print("\n===== Loading and processing test data =====")
    test_path = "datasets/test.csv"
    test_df, test_encounter_ids = pipeline_test(test_path)

    # 確保測試數據的特徵順序
    expected_feature_order = ['apache_4a_hospital_death_prob', 'd1_mbp_min', 'h1_resprate_min', 
                            'd1_temp_max', 'd1_sysbp_noninvasive_max', 'd1_potassium_min', 'ventilated_apache']
    test_df = test_df[expected_feature_order]

    # Save processed test data
    processed_test_path = "datasets/test_processed.csv"
    test_df.to_csv(processed_test_path, index=False)
    print(f"Processed test data saved to {processed_test_path}")

    print("\n===== Making predictions on test data =====")

    # 找出 apache_4a_hospital_death_prob == -1 的樣本
    mask = test_df['apache_4a_hospital_death_prob'] == -1

    # 建立預測結果陣列，初始值為 0
    y_pred_binary_test = np.zeros(len(test_df), dtype=int)

    # 對於 `apache_4a_hospital_death_prob != -1` 的樣本，使用模型預測
    valid_indices = ~mask  # 取反，選擇非 -1 的數據
    if valid_indices.sum() > 0:
        # 取得非 -1 的樣本
        test_df_valid = test_df.loc[valid_indices]

        # 預測機率
        y_pred_proba_test = model.predict_proba(test_df_valid)[:, 1]

        # 根據閾值轉換為二元標籤
        y_pred_binary_test[valid_indices] = (y_pred_proba_test > optimal_threshold).astype(int)

    # 儲存最終預測結果
    if test_encounter_ids is not None:
        save_submission(test_encounter_ids, y_pred_binary_test, 'submission.csv')
    else:
        print("Warning: No encounter IDs found for test data. Cannot create submission file.")
    # Feature importance visualization
    print("\n===== Creating visualizations =====")
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=len(SELECTED_FEATURES))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    
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
    
    print("\nAnalysis complete. Results and visualizations saved.")

if __name__ == "__main__":
    main()