import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import csv
import os
import pickle
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_rn.log"),
        logging.StreamHandler()
    ]
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


class EpisodeDataset(Dataset):
    """Dataset for generating few-shot learning episodes."""
    
    def __init__(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        n_way: int = 2,
        n_shot: int = 5,
        n_query: int = 15,
        episodes_per_epoch: int = 100
    ):
        """
        Initialize EpisodeDataset.
        
        Args:
            features: Feature matrix
            labels: Class labels
            n_way: Number of classes per episode
            n_shot: Number of support examples per class
            n_query: Number of query examples per class
            episodes_per_epoch: Number of episodes to generate per epoch
        """
        self.features = features
        self.labels = labels
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        
        # Group indices by class
        self.label_indices = {}
        for i, label in enumerate(self.labels):
            if label not in self.label_indices:
                self.label_indices[label] = []
            self.label_indices[label].append(i)
            
        # Check if we have enough samples for each class
        for label, indices in self.label_indices.items():
            if len(indices) < (n_shot + n_query):
                logging.warning(f"Class {label} has only {len(indices)} samples, which is less than n_shot({n_shot}) + n_query({n_query})")
    
    def __len__(self):
        return self.episodes_per_epoch
    
    def __getitem__(self, index):
    # For binary classification, we always include both classes (n_way = 2)
        selected_classes = list(self.label_indices.keys())

        support_features = []
        support_labels = []
        query_features = []
        query_labels = []

        # For each class
        for i, cls in enumerate(selected_classes):
            # Get indices for this class
            cls_indices = self.label_indices[cls].copy()
            np.random.shuffle(cls_indices)

            # Split into support and query
            support_idx = cls_indices[:self.n_shot]
            query_idx = cls_indices[self.n_shot:(self.n_shot + self.n_query)]
            print(f"Support indices: {support_idx}")
            print(f"Columns in features: {self.features.columns}")

            # Use .iloc[] to select rows based on indices
            support_features.append(self.features.iloc[support_idx].values)
            support_labels.append(np.full(len(support_idx), i))

            query_features.append(self.features.iloc[query_idx].values)
            query_labels.append(np.full(len(query_idx), i))

        # Convert to numpy arrays and reshape
        support_features = np.vstack(support_features)
        support_labels = np.hstack(support_labels)
        query_features = np.vstack(query_features)
        query_labels = np.hstack(query_labels)

        # Convert to torch tensors
        support_features = torch.FloatTensor(support_features)
        support_labels = torch.LongTensor(support_labels)
        query_features = torch.FloatTensor(query_features)
        query_labels = torch.LongTensor(query_labels)

        return support_features, support_labels, query_features, query_labels

class RelationModule(nn.Module):
    """Relation Module for comparing query and support samples."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize RelationModule.
        
        Args:
            input_size: Size of concatenated feature vector
            hidden_size: Size of hidden layer
        """
        super(RelationModule, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """Forward pass through the relation module."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return torch.sigmoid(out)


class FeatureEncoder(nn.Module):
    """Feature encoder for extracting features from samples."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize FeatureEncoder.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer size
        """
        super(FeatureEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        """Forward pass through the feature encoder."""
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class RelationNetwork(nn.Module):
    """Relation Network for few-shot learning."""
    
    def __init__(self, input_size: int, feature_dim: int = 64, relation_dim: int = 64):
        """
        Initialize RelationNetwork.
        
        Args:
            input_size: Dimension of input features
            feature_dim: Dimension of encoded features
            relation_dim: Dimension of relation module hidden layer
        """
        super(RelationNetwork, self).__init__()
        self.feature_encoder = FeatureEncoder(input_size, feature_dim)
        self.relation_module = RelationModule(feature_dim, relation_dim)
        
    def forward(self, support_features, support_labels, query_features):
        """
        Forward pass through the Relation Network.
        
        Args:
            support_features: Features of support samples
            support_labels: Labels of support samples
            query_features: Features of query samples
            
        Returns:
            Predicted scores for each query sample for each class
        """
        # Get feature embeddings
        support_features = self.feature_encoder(support_features)
        query_features = self.feature_encoder(query_features)
        
        # Get unique class labels
        classes = torch.unique(support_labels)
        n_way = len(classes)
        n_query = query_features.shape[0]
        
        # For each class, calculate relation scores with each query sample
        scores = torch.zeros(n_query, n_way)
        
        for i, cls in enumerate(classes):
            # Get support samples for this class
            class_support_features = support_features[support_labels == cls]
            
            # Calculate prototype (mean) for this class
            prototype = class_support_features.mean(dim=0)
            
            # Replicate prototype for each query sample
            repeated_prototype = prototype.repeat(n_query, 1)
            
            # Concatenate query features with prototype
            pairs = torch.cat((query_features, repeated_prototype), dim=1)
            
            # Compute relation scores
            relation_scores = self.relation_module(pairs)
            scores[:, i] = relation_scores.squeeze()
        
        return scores


class RelationNetworkWrapper:
    """Wrapper for RelationNetwork to provide scikit-learn compatible interface."""
    
    def __init__(
        self, 
        input_size: int,
        feature_dim: int = 64,
        relation_dim: int = 64,
        n_way: int = 2,
        n_shot: int = 5,
        n_query: int = 15,
        episodes_per_epoch: int = 100,
        learning_rate: float = 0.001,
        n_epochs: int = 50,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize RelationNetworkWrapper.
        
        Args:
            input_size: Dimension of input features
            feature_dim: Dimension of encoded features
            relation_dim: Dimension of relation module hidden layer
            n_way: Number of classes per episode
            n_shot: Number of support examples per class
            n_query: Number of query examples per class
            episodes_per_epoch: Number of episodes to generate per epoch
            learning_rate: Learning rate for optimizer
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.relation_dim = relation_dim
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        
        self.model = RelationNetwork(input_size, feature_dim, relation_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # For feature importances
        self._feature_importances = None
    
    # 其他層的評估模式設置
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RelationNetworkWrapper':
        """
        Train the Relation Network.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: The fitted model
        """
        # Create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets for training and validation
        train_dataset = EpisodeDataset(
            X_train, y_train, 
            n_way=self.n_way, 
            n_shot=self.n_shot, 
            n_query=self.n_query,
            episodes_per_epoch=self.episodes_per_epoch
        )
        
        val_dataset = EpisodeDataset(
            X_val, y_val, 
            n_way=self.n_way, 
            n_shot=self.n_shot, 
            n_query=self.n_query,
            episodes_per_epoch=50  # Fewer episodes for validation
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_acc = 0
            
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(train_loader):
                # Move data to device
                support_x = support_x.squeeze(0).to(self.device)
                support_y = support_y.squeeze(0).to(self.device)
                query_x = query_x.squeeze(0).to(self.device)
                query_y = query_y.squeeze(0).to(self.device)
                
                # Forward pass
                relations = self.model(support_x, support_y, query_x)
                
                # Create one-hot encoded targets
                one_hot_targets = torch.zeros_like(relations)
                for i, target in enumerate(query_y):
                    one_hot_targets[i, target] = 1
                
                # Compute loss
                loss = self.criterion(relations, one_hot_targets)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Compute accuracy
                _, predicted = relations.max(1)
                accuracy = (predicted == query_y).float().mean().item()
                
                train_loss += loss.item()
                train_acc += accuracy
            
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_acc = 0
            
            with torch.no_grad():
                for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(val_loader):
                    # Move data to device
                    support_x = support_x.squeeze(0).to(self.device)
                    support_y = support_y.squeeze(0).to(self.device)
                    query_x = query_x.squeeze(0).to(self.device)
                    query_y = query_y.squeeze(0).to(self.device)
                    
                    # Forward pass
                    relations = self.model(support_x, support_y, query_x)
                    
                    # Compute accuracy
                    _, predicted = relations.max(1)
                    accuracy = (predicted == query_y).float().mean().item()
                    
                    val_acc += accuracy
                
                val_acc /= len(val_loader)
            
            logging.info(f"Epoch {epoch+1}/{self.n_epochs}, "
                         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                         f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save the best model
                torch.save(self.model.state_dict(), 'models/best_relation_network.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load the best model
        self.model.load_state_dict(torch.load('models/best_relation_network.pt'))
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def predict(self, X: np.ndarray, support_X: Optional[np.ndarray] = None, 
               support_y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class labels for X.
        
        Args:
            X: Features to predict on
            support_X: Support set features (if None, use training data)
            support_y: Support set labels (if None, use training data)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        proba = self.predict_proba(X, support_X, support_y)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray, support_X: Optional[np.ndarray] = None, 
                     support_y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X: Features to predict on
            support_X: Support set features (if None, use training data)
            support_y: Support set labels (if None, use training data)
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        self.model.eval()
        
        # If support set not provided, use a random subset of training data
        if support_X is None or support_y is None:
            # This is a placeholder - in a real implementation, 
            # you would store some training data during fit()
            raise ValueError("Support set must be provided")
        
        # Convert to torch tensors
        support_X_tensor = torch.FloatTensor(support_X).to(self.device)
        support_y_tensor = torch.LongTensor(support_y).to(self.device)
        query_X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            relations = self.model(support_X_tensor, support_y_tensor, query_X_tensor)
        
        # Return as numpy array
        return relations.cpu().numpy()
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate feature importances using permutation importance.
        
        Args:
            X: Features
            y: Labels
        """
    # Create a small subset for feature importance calculation
        X_subset, _, y_subset, _ = train_test_split(
            X, y, test_size=0.8, random_state=42, stratify=y
        )
    
    # Use a small support set
        support_idx = []
        for label in np.unique(y_subset):
            # Get indices for this class
            class_idx = np.where(y_subset == label)[0]
            # Select n_shot random indices
            selected_idx = np.random.choice(class_idx, self.n_shot, replace=False)
            support_idx.extend(selected_idx)
    
    # Use .iloc to select support_X and support_y from the pandas DataFrame
        support_X = X_subset.iloc[support_idx, :]
        support_y = y_subset.iloc[support_idx]
    
    # Get remaining indices for test set
        test_idx = np.setdiff1d(np.arange(len(X_subset)), support_idx)
        test_X = X_subset.iloc[test_idx,:]
        test_y = y_subset.iloc[test_idx]
    
    # Baseline performance
        self.model.eval()
        with torch.no_grad():
            support_X_tensor = torch.FloatTensor(support_X.values).to(self.device)  # Convert to NumPy array first
            support_y_tensor = torch.LongTensor(support_y.values).to(self.device)  # Convert to NumPy array first
            test_X_tensor = torch.FloatTensor(test_X.values).to(self.device)  # Convert to NumPy array first

            baseline_relations = self.model(support_X_tensor, support_y_tensor, test_X_tensor)
            baseline_pred = torch.argmax(baseline_relations, dim=1).cpu().numpy()
            baseline_acc = (baseline_pred == test_y).mean()
    
    # Calculate importance for each feature
        importances = np.zeros(self.input_size)
    
        for i in range(self.input_size):
            # Create permuted feature
            X_permuted = test_X.copy()  # Make a copy of the data
            # Permute the i-th feature column in X_permuted
            X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i].values)  # Use .iloc to modify the column
            
            # Evaluate on permuted data
            with torch.no_grad():
                test_X_permuted = torch.FloatTensor(X_permuted.values).to(self.device)  # Convert to NumPy array first
                permuted_relations = self.model(support_X_tensor, support_y_tensor, test_X_permuted)
                permuted_pred = torch.argmax(permuted_relations, dim=1).cpu().numpy()
                permuted_acc = (permuted_pred == test_y).mean()
            
            # Importance is the decrease in performance
            importances[i] = baseline_acc - permuted_acc

    # Normalize importances
        importances = np.maximum(importances, 0)  # Only consider features that decrease performance
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)

        self._feature_importances = importances
    def score(self, X_val, y_val):

    # Convert inputs to numpy arrays if they aren't already
        if not isinstance(X_val, np.ndarray):
            X_val = X_val.values
        if not isinstance(y_val, np.ndarray):
            y_val = y_val.values
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # We need support data for prediction
        # Here we'll use a subset of X_val/y_val as support set
        unique_classes = np.unique(y_val)
        support_X = []
        support_y = []
        
        for cls in unique_classes:
            # Get indices for this class
            class_idx = np.where(y_val == cls)[0]
            # Select n_shot random indices (or all if fewer available)
            n_samples = min(len(class_idx), self.n_shot)
            selected_idx = np.random.choice(class_idx, n_samples, replace=False)
            support_X.append(X_val[selected_idx])
            support_y.append(np.full(n_samples, cls))
        
        support_X = np.vstack(support_X)
        support_y = np.hstack(support_y)
        
        # Get predictions using predict method
        y_pred = self.predict(X_val, support_X, support_y)
        
        # Compute accuracy
        accuracy = (y_pred == y_val).mean()
        
        return accuracy
    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances."""
        if self._feature_importances is None:
            raise ValueError("Feature importances not calculated. Call fit() first.")
        return self._feature_importances
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        model_dict = {
            'state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'feature_dim': self.feature_dim,
            'relation_dim': self.relation_dim,
            'n_way': self.n_way,
            'n_shot': self.n_shot,
            'n_query': self.n_query,
            'learning_rate': self.learning_rate,
            'feature_importances': self._feature_importances
        }
        torch.save(model_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'RelationNetworkWrapper':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            RelationNetworkWrapper: Loaded model
        """
        checkpoint = torch.load(path)
        
        # Create instance
        instance = cls(
            input_size=checkpoint['input_size'],
            feature_dim=checkpoint['feature_dim'],
            relation_dim=checkpoint['relation_dim'],
            n_way=checkpoint['n_way'],
            n_shot=checkpoint['n_shot'],
            n_query=checkpoint['n_query'],
            learning_rate=checkpoint['learning_rate']
        )
        
        # Load state dict
        instance.model.load_state_dict(checkpoint['state_dict'])
        
        # Load feature importances
        instance._feature_importances = checkpoint['feature_importances']
        
        return instance


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


def feature_selection_with_permutation(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str],
    support_size: int = 5,
    top_n: int = 30,
    n_iterations: int = 5
) -> List[str]:
    """
    Perform feature selection using permutation importance.
    
    Args:
        X: Feature array
        y: Target array
        feature_names: List of feature names
        support_size: Number of support samples per class
        top_n: Number of top features to select
        n_iterations: Number of iterations for more stable feature importance
        
    Returns:
        List[str]: Selected feature names
    """
    logging.info("Starting feature selection with permutation importance...")
    
    # Split data for feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Get support set (stratified by class)
    support_X = []
    support_y = []
    
    for label in np.unique(y_train):
        # Get indices for this class
        class_idx = np.where(y_train == label)[0]
        # Select support_size random indices
        selected_idx = np.random.choice(class_idx, support_size, replace=False)
        support_X.append(X_train[selected_idx])
        support_y.append(np.full(support_size, label))
    
    support_X = np.vstack(support_X)
    support_y = np.hstack(support_y)
    
    # Create a small model for feature selection
    temp_model = RelationNetwork(
        input_size=X.shape[1],
        feature_dim=32,  # Smaller for faster computation
        relation_dim=32
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Train for a few epochs
    temp_model.train()
    
    # Convert to torch tensors
    support_X_tensor = torch.FloatTensor(support_X).to('cuda' if torch.cuda.is_available() else 'cpu')
    support_y_tensor = torch.LongTensor(support_y).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the same test set for all iterations
    test_idx = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
    test_X = X_test[test_idx]
    test_y = y_test[test_idx]
    test_X_tensor = torch.FloatTensor(test_X).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(10):  # Just a few epochs for feature selection
        # Forward pass
        relations = temp_model(support_X_tensor, support_y_tensor, support_X_tensor)
        
        # Create one-hot encoded targets
        one_hot_targets = torch.zeros_like(relations)
        for i, target in enumerate(support_y_tensor):
            one_hot_targets[i, target] = 1
        
        # Compute loss
        loss = criterion(relations, one_hot_targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Now calculate feature importance with permutation
    temp_model.eval()
    
    # Baseline performance
    with torch.no_grad():
        baseline_relations = temp_model(support_X_tensor, support_y_tensor, test_X_tensor)
        baseline_pred = torch.argmax(baseline_relations, dim=1).cpu().numpy()
        baseline_acc = (baseline_pred == test_y).mean()
    
    # Calculate importance for each feature
    feature_importances = np.zeros(X.shape[1])
    
    for iteration in range(n_iterations):
        logging.info(f"Feature selection iteration {iteration+1}/{n_iterations}")
        for i in range(X.shape[1]):
            # Create permuted feature
            X_permuted = test_X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Evaluate on permuted data
            X_permuted_tensor = torch.FloatTensor(X_permuted).to('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                permuted_relations = temp_model(support_X_tensor, support_y_tensor, X_permuted_tensor)
                permuted_pred = torch.argmax(permuted_relations, dim=1).cpu().numpy()
                permuted_acc = (permuted_pred == test_y).mean()
            
            # Importance is decrease in performance when feature is permuted
            feature_importances[i] += (baseline_acc - permuted_acc) / n_iterations
    
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    selected_features = [feature_names[i] for i in indices[:top_n]]
    
    logging.info(f"Selected top {len(selected_features)} features")
    
    return selected_features


def evaluate_model(model, X_test, y_test, support_X=None, support_y=None):
    """
    Evaluate the model and return performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        support_X: Support set features (needed for RelationNetwork)
        support_y: Support set labels (needed for RelationNetwork)
        
    Returns:
        dict: Performance metrics
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test, support_X, support_y)[:, 1]
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Return results
    return {
        'roc_auc': roc_auc,
        'report': report,
        'confusion_matrix': cm,
        'threshold': optimal_threshold,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_results(results, X_test, y_test, model, feature_names):
    """
    Plot evaluation results.
    
    Args:
        results: Results from evaluate_model
        X_test: Test features
        y_test: Test labels
        model: Trained model
        feature_names: List of feature names
    """
    # Create a figure with subplots
    plt.figure(figsize=(18, 12))
    
    # 1. ROC Curve
    plt.subplot(2, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'ROC curve (area = {results["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 2. Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = results['confusion_matrix']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # 3. Feature Importance
    plt.subplot(2, 2, 3)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances')
    
    # 4. Precision-Recall by Threshold
    plt.subplot(2, 2, 4)
    precision = results['report']['1']['precision']
    recall = results['report']['1']['recall']
    f1 = results['report']['1']['f1-score']
    plt.axvline(x=results['threshold'], color='r', linestyle='--', 
                label=f'Threshold = {results["threshold"]:.3f}')
    plt.text(results['threshold'] + 0.05, 0.8, 
             f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Optimal Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/relation_network_results.png')
    plt.close()


def save_submission(encounter_ids, predictions, filename):
    """
    Save predictions to a CSV file for submission.
    
    Args:
        encounter_ids: Encounter IDs
        predictions: Predicted labels
        filename: Output file name
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['encounter_id', 'hospital_death'])
        for encounter_id, pred in zip(encounter_ids, predictions):
            writer.writerow([encounter_id, pred])
    
    logging.info(f"Submission saved to {filename}")

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
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    input_size = X_train.shape[1]  # 不需要搜索，固定為輸入特徵維度
    feature_dim =  trial.suggest_int('feature_dim', 32, 128)  # 增加搜索範圍
    relation_dim =  trial.suggest_int('relation_dim', 32, 128)  # 增加搜索範圍
    n_way = 2  # 保持固定，因為你的分類問題只有兩類
    n_shot = trial.suggest_int('n_shot', 1, 10) # 允許更多支持樣本
    n_query = trial.suggest_int('n_query', 5, 20)  # 增加查詢樣本數的範圍
    episodes_per_epoch =  trial.suggest_int('episodes_per_epoch', 50, 200)  # 擴大範圍
    learning_rate =  trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)  # 允許更小學習率
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    params = {
        'input_size': input_size,
        'feature_dim': feature_dim,
        'relation_dim': relation_dim,
        'n_way': n_way,
        'n_shot': n_shot ,
        'n_query': n_query,
        'episodes_per_epoch': episodes_per_epoch,
        'learning_rate': learning_rate,
        'n_epochs': 50
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Set up TabNet model
        model = RelationNetworkWrapper(**params)

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
def main():
    """Main execution function for RelationNetwork."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logging.info("Loading and processing training data...")
    train_path = "models/train_augmented.csv"
    
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
    with open('models/encoders_rn.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    X = train_df.drop('hospital_death', axis=1)
    y = train_df['hospital_death']

    # Check for class imbalance
    class_counts = y.value_counts()
    logging.info(f"Class distribution: {class_counts.to_dict()}")
    imbalance_ratio = class_counts[0] / class_counts[1]
    logging.info(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature selection
    logging.info("Starting feature selection...")
    feature_names = X.columns.tolist()
    
    # Check if we have pre-selected features
    if os.path.exists("models/selected_features_rn.txt"):
        logging.info("Loading pre-selected features...")
        with open("models/selected_features_rn.txt", "r") as f:
            selected_features = [line.strip() for line in f]
    else:
        # Feature selection with permutation importance
        selected_features = feature_selection_with_permutation(
            X_train, y_train, 
            feature_names=feature_names, 
            top_n=30,
            n_iterations=3
        )

        # Save selected features for future use
        with open("models/selected_features_rn.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
    
    # Filter X to selected features
    X_train_selected = X_train[:, [feature_names.index(f) for f in selected_features]]
    X_test_selected = X_test[:, [feature_names.index(f) for f in selected_features]]
    
    # Model parameters - these could be optimized with Optuna as in the TabNet example
    '''
    model_params = {
        'input_size': X_train_selected.shape[1],
        'feature_dim': 64,
        'relation_dim': 64,
        'n_way': 2,
        'n_shot': 5,
        'n_query': 15,
        'episodes_per_epoch': 100,
        'learning_rate': 0.001,
        'n_epochs': 50
    }'''
    logging.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize', 
                              study_name='1relationnet_optimization',
                               storage='sqlite:///models/optuna_study.db',
                               load_if_exists=True)
    study.optimize(lambda trial: objective(trial, X_train_selected, y_train), n_trials=50)

    # Log best parameters
    logging.info(f"Best hyperparameters: {study.best_params}")
    logging.info(f"Best score: {study.best_value:.4f}")
    best_params = study.best_params
    #best_params =  {'n_d': 9, 'n_a': 62, 'gamma': 1.0521735750265184, 'n_steps': 8, 'lambda_sparse': 0.0022649452287559066, 'lr': 0.0850686286202479}
    # Train final model with best parameters
    ''''''
    X_train_selected = pd.DataFrame(X_train_selected)  # 將 X_train_selected 轉換為 DataFrame
    y_train = pd.Series(y_train)  # 如果需要，可以將 y_train 轉換為 Series

    #best_params = {'input_size': X_train_selected.shape[1],'feature_dim': 119, 'relation_dim': 118, 'n_shot': 10, 'n_query': 6, 'episodes_per_epoch': 151, 'learning_rate': 0.004857824677314656}
    logging.info("Training Relation Network...")
    model = RelationNetworkWrapper(**best_params)
    model.fit(X_train_selected, y_train)
    
    # Save the model
    model.save('models/best_relation_network.pkl')
    logging.info("Model saved to models/best_relation_network.pkl")
    
    # Get support set for evaluation (stratified by class)
    support_X = []
    support_y = []
    
    for label in np.unique(y_train):
    # Get indices for this class
        class_idx = np.where(y_train == label)[0]
        # Select n_shot random indices
        selected_idx = np.random.choice(class_idx, best_params['n_shot'], replace=False)
        support_X.append(X_train_selected.iloc[selected_idx, :].values)  # 使用 .iloc 來選擇行
        support_y.append(np.full(best_params['n_shot'], label))

    support_X = np.vstack(support_X)
    support_y = np.hstack(support_y)
    
    # Evaluate the model
    results = evaluate_model(model, X_test_selected, y_test, support_X, support_y)
    logging.info(f"ROC AUC on test set: {results['roc_auc']:.4f}")
    logging.info(f"Classification report:\n{classification_report(y_test, results['y_pred'])}")
    
    # Plot results
    plot_results(results, X_test_selected, y_test, model, selected_features)
    
    # Process test data for submission
    logging.info("Loading and processing test data...")
    test_path = "datasets/test.csv"
    
    try:
        # Process test data with the same transformations
        test_df, test_encounter_ids, _ = pipeline(
            test_path, 
            encoders=encoders
        )
        
        # Filter to selected features
        for feature in selected_features:
            if feature not in test_df.columns:
                test_df[feature] = 0  # Add missing columns with zeros
        
        test_df_selected = test_df[selected_features].values
        
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        logging.error("Please ensure the test dataset file exists at the specified path.")
        return

    # Make predictions on test data
    logging.info("Generating predictions for test data...")
    y_pred_proba_test = model.predict_proba(test_df_selected, support_X, support_y)[:, 1]
    y_pred_binary_test = (y_pred_proba_test > results['threshold']).astype(int)

    # Save submission
    if test_encounter_ids is not None:
        save_submission(test_encounter_ids, y_pred_binary_test, 'results/rn_submission.csv')
    else:
        logging.error("Error: encounter_ids not found in test data.")

    logging.info("Process completed successfully!")


if __name__ == "__main__":
    main()