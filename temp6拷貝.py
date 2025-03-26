import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import csv
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.special import softmax
from scipy.special import expit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Suppress TabNet UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet")


class TabNetClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for TabNetClassifier to make it compatible with scikit-learn API."""
    
    def __init__(
        self, 
        n_d: int = 8, 
        n_a: int = 8, 
        optimizer_params: Optional[Dict] = None, 
        gamma: float = 1.3, 
        n_steps: int = 3, 
        lambda_sparse: float = 0.001,
        max_epochs: int = 100, 
        patience: int = 15,
        verbose: int = 0,
        device_name: str = 'auto'
    ):
        """
        Initialize TabNetClassifierWrapper.
        
        Args:
            n_d: Width of the decision prediction layer. Shared across all steps.
            n_a: Width of the attention embedding for each step.
            optimizer_params: Parameters for the optimizer (default: {"lr": 0.02})
            gamma: Scaling factor for attention updates.
            n_steps: Number of sequential attention steps.
            lambda_sparse: Strength of the sparsity regularization.
            max_epochs: Maximum number of epochs for training.
            patience: Number of epochs for early stopping.
            verbose: Verbosity level (0, 1, or 2).
            device_name: 'cpu', 'cuda', or 'auto' to automatically select device.
        """
        self.n_d = n_d
        self.n_a = n_a
        self.optimizer_params = optimizer_params or {"lr": 0.02}
        self.gamma = gamma
        self.n_steps = n_steps
        self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.device_name = device_name
        self.model = None
        self._estimator_type = "classifier"  # For sklearn compatibility

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'TabNetClassifierWrapper':
        """
        Train TabNet classifier.
        
        Args:
            X: Training features
            y: Target variable
            
        Returns:
            self: The fitted model
        """
        # Convert inputs to numpy arrays if they're pandas objects
        X_array = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y_array = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
            
        # Create validation set for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
        )
        
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            optimizer_params=self.optimizer_params,
            gamma=self.gamma,
            n_steps=self.n_steps,
            lambda_sparse=self.lambda_sparse,
            verbose=self.verbose,
            device_name=self.device_name,
            seed=42
        )
        
        self.model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            max_epochs=self.max_epochs, 
            patience=self.patience
        )
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Return class predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Class predictions
        """
        self._check_is_fitted()
        X_array = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Return class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Class probabilities
        """
        self._check_is_fitted()
        X_array = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        return self.model.predict_proba(X_array)

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate ROC AUC score.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            float: ROC AUC score
        """
        X_array = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y_array = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
        y_pred_proba = self.predict_proba(X_array)[:, 1]  # Get positive class probability
        return roc_auc_score(y_array, y_pred_proba)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'TabNetClassifierWrapper':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            TabNetClassifierWrapper: Loaded model
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances from the underlying model."""
        self._check_is_fitted()
        return self.model.feature_importances_


def save_submission(encounter_ids: pd.Series, predictions: np.ndarray, filename: str) -> None:
    """
    Save predictions to a CSV file for submission.
    
    Args:
        encounter_ids: Series of encounter IDs
        predictions: Array of predictions
        filename: Output file path
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['encounter_id', 'hospital_death'])
        for i in range(len(predictions)):
            writer.writerow([encounter_ids.iloc[i], int(predictions[i])])
    logging.info(f"Submission saved to {filename}")


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


def objective(trial, X_train, y_train):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        
    Returns:
        float: Average validation score
    """
    # Let Optuna explore different hyperparameters
    n_d = trial.suggest_int('n_d', 8, 64)
    n_a = trial.suggest_int('n_a', 8, 64)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    lambda_sparse = trial.suggest_float('lambda_sparse', 0.0001, 0.1, log=True)
    lr = trial.suggest_float('lr', 0.005, 0.1, log=True)

    # Set up k-fold cross-validation
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
        'patience': 10,
        'verbose': 0
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Set up TabNet model
        model = TabNetClassifierWrapper(**params)

        # Train model
        model.fit(X_cv_train, y_cv_train)
        
        # Evaluate model
        score = model.score(X_cv_val, y_cv_val)
        cv_scores.append(score)
        logging.debug(f"Fold {fold+1}: ROC AUC = {score:.4f}")

    # Calculate average cross-validation score
    avg_score = np.mean(cv_scores)
    
    # Log parameters and score
    logging.info(f"Trial {trial.number}: params={trial.params}, score={avg_score:.4f}")
    
    return avg_score


def feature_selection_with_tabnet(
    data: pd.DataFrame, 
    target: pd.Series, 
    features: List[str], 
    max_features: int = 50, 
    test_size: float = 0.2, 
    early_stopping_rounds: int = 3, 
    min_improvement: float = 0.001, 
    random_state: int = 42
) -> Tuple[List[str], List[float]]:
    """
    Perform greedy feature selection with early stopping.
    
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
        Tuple containing:
        - List of selected features
        - List of scores for each selection round
    """
    # Create a validation set to prevent overfitting during feature selection
    X_train, X_val, y_train, y_val = train_test_split(
        data, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    selected_features = []
    score_history = []
    current_best_score = 0
    no_improvement_count = 0

    # Keep track of features to consider
    remaining_features = set(features)

    for i in range(min(max_features, len(features))):
        best_feature = None
        best_score = current_best_score
        best_improvement = 0

        logging.info(f"Selection round {i+1}/{min(max_features, len(features))}")
        
        # Evaluate remaining features
        for feature in list(remaining_features):
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
            
            logging.debug(f"Feature: {feature}, Score: {trial_score:.4f}, Improvement: {improvement:.4f}")
            
            if improvement > best_improvement:
                best_score = trial_score
                best_feature = feature
                best_improvement = improvement

        # Check if we found a feature with sufficient improvement
        if best_improvement > min_improvement and best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            score_history.append(best_score)
            current_best_score = best_score
            no_improvement_count = 0
            logging.info(f"Selected feature: {best_feature}, Score: {current_best_score:.4f}, Improvement: {best_improvement:.4f}")
        else:
            no_improvement_count += 1
            logging.info(f"No significant improvement found (best: {best_improvement:.4f})")
            
            # Check early stopping condition
            if no_improvement_count >= early_stopping_rounds:
                logging.info(f"Stopping early after {no_improvement_count} rounds without significant improvement")
                break

    # Plot feature selection progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(score_history) + 1), score_history, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('ROC AUC Score')
    plt.title('Feature Selection Progress')
    plt.grid(True)
    plt.savefig('results/feature_selection_progress.png')
    plt.close()

    return selected_features, score_history


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Find the optimal threshold for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        float: Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # Find the optimal point on the ROC curve that maximizes sensitivity-specificity
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def evaluate_model(
    model: TabNetClassifierWrapper, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    threshold: Optional[float] = None
) -> Dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold (if None, optimal threshold is found)
        
    Returns:
        Dict: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_test, y_pred_proba)
        logging.info(f"Optimal threshold: {threshold:.4f}")
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    results = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'mse': mean_squared_error(y_test, y_pred_binary),
        'mae': mean_absolute_error(y_test, y_pred_binary),
        'threshold': threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred_binary)
    }
    
    # Log performance metrics
    logging.info(f"Test ROC AUC: {results['roc_auc']:.4f}")
    logging.info(f"Test MSE: {results['mse']:.4f}")
    logging.info(f"Test MAE: {results['mae']:.4f}")
    logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred_binary))
    
    return results


def plot_results(results: Dict, X_test: pd.DataFrame, y_test: pd.Series, 
                model: TabNetClassifierWrapper, selected_features: List[str]) -> None:
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        X_test: Test features
        y_test: Test target
        model: Trained model
        selected_features: List of selected features
    """
    os.makedirs('results', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    conf_matrix = results['confusion_matrix']
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

    # Plot feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
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
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
    plt.savefig('results/roc_curve.png')
    plt.close()


def main():
    """Main execution function."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logging.info("Loading and processing training data...")
    train_path = "datasets/train_augmented.csv"
    
    try:
        train_df, train_encounter_ids, encoders = pipeline(train_path)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        logging.error("Please ensure the dataset file exists at the specified path.")
        return

    if 'hospital_death' not in train_df.columns:
        logging.error("Error: 'hospital_death' column not found in the data.")
        return

    # Save encoders for future use
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info("Starting feature selection...")
    # Check if we have pre-selected features
    if os.path.exists("models/selected_features.txt"):
        logging.info("Loading pre-selected features...")
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
        with open("models/selected_features.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Hyperparameter optimization with Optuna
    logging.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize', 
                               study_name='1tabnet_optimization',
                               storage='sqlite:///models/optuna_study.db',
                               load_if_exists=True)
    study.optimize(lambda trial: objective(trial, X_train_selected, y_train), n_trials=50)

    # Log best parameters
    #logging.info(f"Best hyperparameters: {study.best_params}")
    #logging.info(f"Best score: {study.best_value:.4f}")
    #best_params =  {'n_d': 9, 'n_a': 62, 'gamma': 1.0521735750265184, 'n_steps': 8, 'lambda_sparse': 0.0022649452287559066, 'lr': 0.0850686286202479}
    # Train final model with best parameters
    best_params = study.best_params.copy()
    best_lr = best_params.pop("lr")  # Extract lr to avoid initialization error
    
    logging.info("Training final model with best parameters...")
    best_model = TabNetClassifierWrapper(
        **best_params, 
        optimizer_params={"lr": best_lr},
        verbose=1  # Show some training progress
    )
    best_model.fit(X_train_selected, y_train)

    # Save the trained model
    best_model.save('models/best_model.pkl')
    logging.info("Model saved to models/best_model.pkl")

    # Evaluate on test set
    results = evaluate_model(best_model, X_test_selected, y_test)
    
    # Plot results
    plot_results(results, X_test_selected, y_test, best_model, selected_features)
    
    # Process test data for submission
    logging.info("Loading and processing test data...")
    test_path = "datasets/test.csv"
    
    try:
        # Process test data with the same transformations
        test_df, test_encounter_ids, _ = pipeline(
            test_path, 
            encoders=encoders, 
            selected_features=selected_features
        )
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        logging.error("Please ensure the test dataset file exists at the specified path.")
        return

    # Make predictions
    logging.info("Generating predictions for test data...")
      # 應該是 float32 或 float64
    y_pred_proba_test = best_model.predict_proba(test_df)[:, 1]
    y_pred_binary_test = (y_pred_proba_test > results['threshold']).astype(int)

    # Save submission
    if test_encounter_ids is not None:
        save_submission(test_encounter_ids, y_pred_binary_test, 'results/submission.csv')
    else:
        logging.error("Error: encounter_ids not found in test data.")

    logging.info("Process completed successfully!")


if __name__ == "__main__":
    main()