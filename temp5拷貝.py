import numpy as np
import optuna
import pandas as pd
import logging 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import csv
import os
import pickle
import warnings
from sklearn.model_selection import KFold
from typing import Union, Dict, List, Tuple, Optional
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet")

class TabNetClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_d=8, n_a=8, optimizer_params=None, gamma=1.3, n_steps=3, lambda_sparse=0.001,
                 max_epochs=100, patience=15):
        self.n_d = n_d
        self.n_a = n_a
        self.optimizer_params = optimizer_params or {"lr": 0.02}
        self.gamma = gamma
        self.n_steps = n_steps
        self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs
        self.patience = patience
        self.model = None
        self._estimator_type = "classifier"  # Explicitly set estimator type for sklearn

    def fit(self, X, y):
        """Train TabNet classifier"""
        # Convert inputs to numpy arrays if they're pandas objects
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
            
        # Create validation set for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            optimizer_params=self.optimizer_params,
            gamma=self.gamma,
            n_steps=self.n_steps,
            lambda_sparse=self.lambda_sparse,
            verbose=0,
            seed=42
        )
        
        self.model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            max_epochs=self.max_epochs, 
            patience=self.patience
        )
        return self

    def predict(self, X):
        """Return class labels"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return class probabilities for ROC AUC scoring"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
            
        # Return probabilities for both classes (required for sklearn scorers)
        return self.model.predict_proba(X)

    def score(self, X, y):
        """Calculate ROC AUC score"""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        y_pred_proba = self.predict_proba(X)[:, 1]  # Get positive class probability
        return roc_auc_score(y, y_pred_proba)
    
    def save(self, path):
        """Save the model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load a model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def save_submission(encounter_ids, predictions, filename):
    """Save predictions to a CSV file for submission."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['encounter_id', 'hospital_death'])
        for i in range(len(predictions)):
            writer.writerow([encounter_ids.iloc[i], int(predictions[i])])
    print(f"Submission saved to {filename}")


def handle_missing_values_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Handle missing values using KNN imputation for numerical features.
    Simple imputation for categorical features.
    
    Args:
        df: DataFrame with missing values
        n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        DataFrame with imputed values
    """
    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Handle numerical features with KNN imputation
    if num_cols:
        # Create a copy to avoid modifying original data
        df_numeric = df[num_cols].copy()
        
        # Initialize KNN imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        # Apply KNN imputation
        imputed_data = imputer.fit_transform(df_numeric)
        
        # Replace original data with imputed values
        df[num_cols] = imputed_data
    
    # For categorical features, use mode imputation
    for col in cat_cols:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        
    return df

def extract_features(df: pd.DataFrame, target_col: str = None, 
                    method: str = 'statistical', n_features: int = 20) -> pd.DataFrame:
    """
    Extract features using statistical methods, PCA, or clustering.
    
    Args:
        df: DataFrame with features
        target_col: Target column name (if available)
        method: Feature extraction method ('statistical', 'pca', 'clustering')
        n_features: Number of features to keep
        
    Returns:
        DataFrame with extracted features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Only work with numerical columns
    num_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from numeric columns if present
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    
    if len(num_cols) == 0:
        return result_df
    
    # Get numeric data
    X = result_df[num_cols]
    
    # Standardize features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature extraction based on method
    if method == 'statistical' and target_col and target_col in result_df.columns:
        # Get target
        y = result_df[target_col]
        
        # Select K best features
        selector = SelectKBest(mutual_info_classif, k=min(n_features, len(num_cols)))
        selector.fit(X_scaled, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_features = [num_cols[i] for i in selected_indices]
        
        # Create DataFrame with only selected features plus target
        keep_cols = selected_features.copy()
        if target_col:
            keep_cols.append(target_col)
        
        # Also keep categorical columns
        cat_cols = result_df.select_dtypes(include=['object']).columns.tolist()
        keep_cols.extend(cat_cols)
        
        return result_df[keep_cols]
        
    elif method == 'pca':
        # Apply PCA
        n_components = min(n_features, len(num_cols))
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA components
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=result_df.index)
        
        # Get categorical columns and target if available
        cat_cols = result_df.select_dtypes(include=['object']).columns.tolist()
        keep_cols = cat_cols.copy()
        if target_col:
            keep_cols.append(target_col)
        
        # Combine PCA components with categorical columns and target
        for col in keep_cols:
            pca_df[col] = result_df[col]
            
        return pca_df
        
    elif method == 'clustering':
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=min(10, len(result_df)))
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster as a feature
        result_df['cluster'] = clusters
        
        return result_df
        
    else:
        # Return original if no valid method specified
        return result_df

def pipeline(
    df_or_path: Union[str, pd.DataFrame],
    encoders: Optional[Dict] = None,
    selected_features: Optional[List[str]] = None,
    advanced_processing: bool = True,
    imputation_method: str = 'knn',
    feature_extraction: str = None,
    n_features: int = 20
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    Process dataframe or load from path, handle missing values and encode categorical features.
    
    Args:
        df_or_path: DataFrame or path to CSV file
        encoders: Dictionary of category encoders for consistent transforms
        selected_features: List of selected features to include
        advanced_processing: Whether to apply advanced data processing
        imputation_method: Method for imputation ('simple' or 'knn')
        feature_extraction: Method for feature extraction (None, 'statistical', 'pca', 'clustering')
        n_features: Number of features to keep if using feature extraction
        
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

    # Check if target column exists
    target_col = 'hospital_death' if 'hospital_death' in df.columns else None

    # Drop identifier columns if they exist
    id_columns = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
    df = df.drop(columns=[col for col in id_columns if col in df.columns])

    # Handle missing values
    if advanced_processing and imputation_method == 'knn':
        df = handle_missing_values_knn(df)
    else:
        # Original simple imputation
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

    # Apply feature extraction if requested
    if advanced_processing and feature_extraction:
        df = extract_features(df, target_col=target_col, 
                             method=feature_extraction, n_features=n_features)
    
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

# Original function (assumed to be defined elsewhere)
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Simple missing value handling for backwards compatibility."""
    # Fill numeric columns with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    
    return df

def objective(trial , X_train_selected , y_train):
    # 讓 Optuna 探索不同的超參數
    n_d = trial.suggest_int('n_d', 8, 32)
    n_a = trial.suggest_int('n_a', 8, 32)
    gamma = trial.suggest_float('gamma', 1.2, 1.8)
    n_steps = trial.suggest_int('n_steps', 3, 7)
    lambda_sparse = trial.suggest_float('lambda_sparse', 0.001, 0.01, log = True)
    lr = trial.suggest_categorical('lr', [0.1, 0.05, 0.01, 0.07])

    # 設定 K 折交叉驗證
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    params = {
                                'n_d': n_d,
                                'n_a': n_a,
                                'gamma': gamma,
                                'n_steps': n_steps,
                                'lambda_sparse': lambda_sparse,
                                'optimizer_params': {"lr": lr},
                                'max_epochs': 50,
                                'patience': 10
                            }
    # 進行交叉驗證
    for train_idx, val_idx in kf.split(X_train_selected):
        X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        '''
        params = {
                                'n_d': n_d,
                                'n_a': n_a,
                                'gamma': gamma,
                                'n_steps': n_steps,
                                'lambda_sparse': lambda_sparse,
                                'optimizer_params': {"lr": lr},
                                'max_epochs': 50,
                                'patience': 10
                            }
        ''' 
        # 設定 TabNet 模型
        model = TabNetClassifierWrapper(**params)

        # 訓練模型
        model.fit(X_cv_train.values, y_cv_train.values)
        
        # 評估模型
        score = model.score(X_cv_val.values, y_cv_val.values)
        cv_scores.append(score)

    # 計算平均交叉驗證分數
    avg_score = np.mean(cv_scores)
    
    # 記錄參數與分數
    with open("models/param.txt", "a") as f:
        f.write(f"Params: lr={lr}, n_d={n_d}, n_a={n_a}, gamma={gamma}, n_steps={n_steps}, lambda={lambda_sparse}, Score: {avg_score:.4f}\n")

    return avg_score
def feature_selection_with_tabnet(data, target, features, max_features=50, test_size=0.2, 
                                 early_stopping_rounds=3, min_improvement=0.001, random_state=42):
    """
    Perform greedy feature selection with early stopping
    
    Args:
        data: DataFrame containing features
        target: Series containing target variable
        features: List of features to consider
        max_features: Maximum number of features to select
        test_size: Proportion of data to use for validation
        early_stopping_rounds: Stop if no improvement after this many rounds
        min_improvement: Minimum improvement required to count as improvement
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected features and score history
    """
    # Create a validation set to prevent overfitting during feature selection
    X_train, X_val, y_train, y_val = train_test_split(
        data, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    selected_features = []
    score_history = []
    current_best_score = 0
    no_improvement_count = 0

    for i in range(min(max_features, len(features))):
        best_feature = None
        best_score = current_best_score
        best_improvement = 0

        print(f"Selection round {i+1}/{min(max_features, len(features))}")
        
        # Evaluate remaining features
        for feature in features:
            if feature in selected_features:
                continue

            trial_features = selected_features + [feature]
            X_train_trial = X_train[trial_features]
            X_val_trial = X_val[trial_features]

            # Train TabNet with early stopping
            model = TabNetClassifier(verbose=0, seed=random_state)
            model.fit(
                X_train_trial.values, y_train.values,
                eval_set=[(X_val_trial.values, y_val.values)],
                max_epochs=25,
                patience=10
            )
            
            # Evaluate on validation set
            y_pred = model.predict_proba(X_val_trial.values)[:, 1]
            trial_score = roc_auc_score(y_val, y_pred)
            improvement = trial_score - current_best_score
            
            if improvement > best_improvement:
                best_score = trial_score
                best_feature = feature
                best_improvement = improvement

        # Check if we found a feature with sufficient improvement
        if best_improvement > min_improvement:
            selected_features.append(best_feature)
            score_history.append(best_score)
            current_best_score = best_score
            no_improvement_count = 0
            print(f"Selected feature: {best_feature}, Score: {current_best_score:.4f}, Improvement: {best_improvement:.4f}")
        else:
            no_improvement_count += 1
            print(f"No significant improvement found (best: {best_improvement:.4f})")
            
            # Check early stopping condition
            if no_improvement_count >= early_stopping_rounds:
                print(f"Stopping early after {no_improvement_count} rounds without significant improvement")
                break

    # Plot feature selection progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(score_history) + 1), score_history, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('ROC AUC Score')
    plt.title('Feature Selection Progress')
    plt.grid(True)
    plt.savefig('feature_selection_progress.png')
    plt.close()

    return selected_features, score_history
def grid_search(X_train_selected , y_train ):
    param_grid = {
        'n_d': [8, 16, 32],
        'n_a': [8, 16, 32],
        'gamma': [1.2, 1.5, 1.8],
        'n_steps': [3, 5, 7],
        'lambda_sparse': [0.001, 0.01],
        'lr': [0.1 , 0.05 , 0.01,0.07]
    }
    
    # Manual grid search
    best_params = None
    best_score = 0
    
    # Split data for cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for n_d in param_grid['n_d']:
        for n_a in param_grid['n_a']:
            for gamma in param_grid['gamma']:
                for n_steps in param_grid['n_steps']:
                    for lambda_sparse in param_grid['lambda_sparse']:
                        for lr in param_grid['lr']:
                            params = {
                                'n_d': n_d,
                                'n_a': n_a,
                                'gamma': gamma,
                                'n_steps': n_steps,
                                'lambda_sparse': lambda_sparse,
                                'optimizer_params': {"lr": lr},
                                'max_epochs': 50,
                                'patience': 10
                            }
                        # Cross-validation scores
                        cv_scores = []
                        for train_idx, val_idx in kf.split(X_train_selected):
                            X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
                            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                            
                            model = TabNetClassifierWrapper(**params)
                            model.fit(X_cv_train, y_cv_train)
                            
                            # Score on validation set
                            score = model.score(X_cv_val, y_cv_val)
                            cv_scores.append(score)
                        
                        # Average score across folds
                        avg_score = np.mean(cv_scores)
                        print(f"Params: lr = {lr}, n_d={n_d}, n_a={n_a}, gamma={gamma}, n_steps={n_steps}, lambda={lambda_sparse}, Score: {avg_score:.4f}")
                        with open("models/param.txt", "a") as f:
                            f.write(f"Params:lr = {lr}, n_d={n_d}, n_a={n_a}, gamma={gamma}, n_steps={n_steps}, lambda={lambda_sparse}, Score: {avg_score:.4f}\n")
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = params
                            
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    with open("models/param.txt", "a") as f:
        f.write(f"Best parameters: {best_params}")
        f.write(f"Best score: {best_score:.4f}")
    return(best_params)
    

def train_tabnet_without_grid_search(X_train, y_train, X_test, y_test, params=None):
    """Train a TabNet model without using GridSearchCV"""
    default_params = {
        'n_d': 16, 
        'n_a': 16,
        'optimizer_params': {"lr": 0.02},
        'gamma': 1.5,
        'n_steps': 5,
        'lambda_sparse': 0.001,
        'max_epochs': 100,
        'patience': 15
    }
    
    # Use provided params if any, otherwise use defaults
    params = params or default_params
    
    # Create and train the model
    model = TabNetClassifierWrapper(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model trained with ROC AUC: {roc_auc:.4f}")
    
    return model, roc_auc


def find_optimal_threshold(y_true, y_pred_proba):
    """Find the optimal threshold for binary classification"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # Find the optimal point on the ROC curve that maximizes sensitivity-specificity
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Loading and processing training data...")
    train_path = "datasets/train.csv"
    
    try:
        train_df, train_encounter_ids, encoders = pipeline(train_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset file exists at the specified path.")
        return

    if 'hospital_death' not in train_df.columns:
        print("Error: 'hospital_death' column not found in the data.")
        return

    # Save encoders for future use
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    X = train_df.drop('hospital_death', axis=1)
    y = train_df['hospital_death']

    # Check for class imbalance
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
    
    # Handle remaining categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Starting feature selection...")
    # Check if we have pre-selected features
    if os.path.exists("models/selected_features.txt"):
        print("Loading pre-selected features...")
        with open("models/selected_features.txt", "r") as f:
            selected_features = [line.strip() for line in f]
    else:
        # Improved feature selection with early stopping
        selected_features, score_history = feature_selection_with_tabnet(
            X_train, y_train, 
            features=X_train.columns.tolist(), 
            max_features=50,
            early_stopping_rounds=3
        )

        # Save selected features for future use
        os.makedirs('models', exist_ok=True)
        with open("models/selected_features.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # OPTION 1: Try a simplified approach without GridSearchCV
    '''
    print("Training TabNet model with default parameters...")
    best_model, best_score = train_tabnet_without_grid_search(
        X_train_selected, y_train, X_test_selected, y_test
    )
    '''
    
    # OPTION 2: If you want to try GridSearchCV still, use manual cross-validation instead
    # This code is commented out as it's an alternative approach
    
    # Define parameter grid
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial :objective(trial , X_test_selected, y_train), n_trials=50)  # 設定 50 次搜尋

# 輸出最佳參數
    print("Best hyperparameters: ", study.best_params)
    print("Best score: ", study.best_value)
    with open("models/param.txt", "a") as f:
        f.write(f"Best parameters: {study.best_params}\n")
        f.write(f"Best score: {study.best_value:.4f}\n")
    #best_params = grid_search(X_test_selected,y_train)
    # Train final model with best parameters
    best_params = study.best_params.copy()
    best_lr = best_params.pop("lr")  # 取出 lr，避免 `__init__()` 出錯

    best_model = TabNetClassifierWrapper(
    **best_params, 
    optimizer_params={"lr": best_lr}  # 設定 optimizer_params
)
    best_model.fit(X_train_selected, y_train)

    # Save the trained model
    best_model.save('models/best_model.pkl')

    # Evaluate on test set
    y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Apply optimal threshold
    y_pred_binary = (y_pred_proba > optimal_threshold).astype(int)

    # Calculate metrics
    roc = roc_auc_score(y_test, y_pred_proba)
    mse = mean_squared_error(y_test, y_pred_binary)
    mae = mean_absolute_error(y_test, y_pred_binary)

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation ROC AUC: {roc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Survived', 'Died']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    # Plot feature importance
    if hasattr(best_model.model, 'feature_importances_'):
        feature_importance = best_model.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importance
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc:.3f})')
    plt.savefig('results/roc_curve.png')
    plt.close()

    # Process test data for submission
    print("Loading and processing test data...")
    test_path = "datasets/test.csv"
    
    try:
        # Load encoders if needed
        if os.path.exists('models/encoders.pkl'):
            with open('models/encoders.pkl', 'rb') as f:
                encoders = pickle.load(f)
        
        # Process test data with the same transformations
        test_df, test_encounter_ids, _ = pipeline(test_path, encoders=encoders, selected_features=selected_features)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the test dataset file exists at the specified path.")
        return

    # Make predictions
    y_pred_proba_test = best_model.predict_proba(test_df)[:, 1]
    #y_pred_binary_test = (y_pred_proba_test > optimal_threshold).astype(int)

    # Save submission
    if test_encounter_ids is not None:
        save_submission(test_encounter_ids, y_pred_proba_test, 'results/submission.csv')
    else:
        print("Error: encounter_ids not found in test data.")

    print("Process completed successfully!")


if __name__ == "__main__":
    main()