import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import optuna
from tqdm import tqdm
import random

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')

MODEL_DIR = os.path.join(ROOT_DIR, 'nn_model_pytorch_v4_varol') # New folder
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'nn_alpha_model.pt')
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'nn_preprocessor.joblib')

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50
EARLY_STOPPING_PATIENCE = 15
N_TRIAL_EPOCHS = 20
N_FINAL_EPOCHS = 200
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# --- 1. SETUP ---

def set_random_seed(seed=42):
    """Set seeds for reproducibility across all libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (Explicitly forcing model and data to this device)")

# --- 2. DATA HANDLING ---

class TabularDataset(Dataset):
    """Custom PyTorch Dataset for preprocessed tabular data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_preprocessor(X_train):
    """
    Builds the "Varol" style ColumnTransformer.
    It imputes with zero and adds a binary indicator for ALL continuous features.
    """
    print("Building 'Varol' style preprocessing pipeline...")
    
    # 1. Define feature groups
    categorical_features_list = ['dimension', 'grid_size']
    
    # Get all other columns as continuous
    continuous_features_list = [
        col for col in X_train.columns if col not in categorical_features_list
    ]

    print(f"  Found {len(continuous_features_list)} continuous features.")
    print(f"  Found {len(categorical_features_list)} categorical features.")

    # 2. Create pipelines
    
    # --- THIS IS THE "VAROL" METHOD ---
    # We create a pipeline that does two things:
    # 1. Impute(0): Fills all NaNs with 0.
    # 2. add_indicator=True: Creates a *new binary column* for every
    #    feature that had a NaN, marking it as 'missing'.
    # 3. Scale: Scales all columns (both original and new indicator columns)
    
    continuous_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Combine in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipe, continuous_features_list),
            ('categorical', categorical_pipe, categorical_features_list)
        ],
        remainder='drop' # Explicitly drop any columns we didn't list
    )
    
    # Fit the preprocessor
    preprocessor.fit(X_train)
    
    # Calculate final number of features
    n_features = preprocessor.transform(X_train).shape[1]
    
    return preprocessor, n_features

def load_and_preprocess_main(data_path):
    """
    Main data loading function.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)

    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha_scaled'] = df['alpha'].clip(1.0, 2.0) - 1.0
    
    y = df['alpha_scaled']

    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'alpha_scaled', 'split', 'distribution_type'
    ]
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    
    # THIS FIX is CRITICAL
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    train_mask = (df['split'] == 'train')
    val_mask = (df['split'] == 'val')
    test_mask = (df['split'] == 'test')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_full, y_train_full

# --- 3. PYTORCH MLP MODEL ---

class MLP(nn.Module):
    """Simple MLP with dynamic layers and sigmoid output."""
    def __init__(self, n_features_in, n_layers, n_units, dropout_rate, activation):
        super(MLP, self).__init__()
        layers = []
        in_features = n_features_in
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 4. TRAINING & VALIDATION LOOPS ---

def train_one_epoch(model, loader, criterion, optimizer):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_model(model, loader, criterion):
    """Runs validation on the model."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_loss = val_loss / len(loader)
    return avg_loss

# --- 5. OPTUNA HYPERPARAMETER TUNING ---

def objective(trial, X_train_tf, y_train, X_val_tf, y_val, n_features_in):
    """Optuna objective function."""
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_layers = trial.suggest_int('n_layers', 2, 4)
    n_units = trial.suggest_int('n_units', 64, 256, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    train_dataset = TabularDataset(X_train_tf, y_train)
    val_dataset = TabularDataset(X_val_tf, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(n_features_in, n_layers, n_units, dropout_rate, activation).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    best_val_loss = np.inf
    for epoch in range(N_TRIAL_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_model(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if np.isnan(val_loss):
            raise optuna.exceptions.TrialPruned()
            
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_loss

# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    set_random_seed(RANDOM_STATE)
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        print(f"Please run this script from the main project directory (c:/TSP_ND_ML_Project/).")
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print("--- 1. Loading and Preprocessing Data ---")
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            X_train_full, y_train_full
        ) = load_and_preprocess_main(DATA_FILE)

        # Build the new preprocessor
        preprocessor, n_features = build_preprocessor(X_train)
        
        print(f"Saving preprocessor to {PREPROCESSOR_FILE}...")
        joblib.dump(preprocessor, PREPROCESSOR_FILE)
        
        print("Transforming data splits...")
        X_train_tf = preprocessor.transform(X_train)
        X_val_tf = preprocessor.transform(X_val)
        X_test_tf = preprocessor.transform(X_test)
        X_train_full_tf = preprocessor.transform(X_train_full)
        
        print(f"Data transformed. New input feature shape: {n_features}")
        
        print(f"\n--- 2. Running Optuna Hyperparameter Tuning ({OPTUNA_N_TRIALS} trials) ---")
        objective_func = lambda trial: objective(
            trial, X_train_tf, y_train, X_val_tf, y_val, n_features
        )
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
        
        if study.best_value == np.inf or np.isnan(study.best_value):
            print("\n\nCRITICAL ERROR: Optuna tuning failed. All trials resulted in 'inf' or 'nan'.")
            print("This indicates a persistent problem with the data or model architecture.")
        else:
            print(f"\nOptuna tuning complete. Best Validation MSE: {study.best_value:.6f}")
            print("Best parameters found:")
            print(study.best_params)

            print("\n--- 3. Training Final Model ---")
            best_params = study.best_params
            
            train_full_dataset = TabularDataset(X_train_full_tf, y_train_full)
            val_size = int(len(train_full_dataset) * VALIDATION_SPLIT)
            train_size = len(train_full_dataset) - val_size
            train_data, val_data = random_split(train_full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_data, batch_size=best_params["batch_size"], shuffle=False)

            final_model = MLP(
                n_features_in=n_features,
                n_layers=best_params['n_layers'],
                n_units=best_params['n_units'],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation']
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = getattr(optim, best_params["optimizer"])(
                final_model.parameters(), lr=best_params["learning_rate"]
            )

            best_val_loss = np.inf
            epochs_no_improve = 0
            best_model_state = final_model.state_dict()

            for epoch in range(N_FINAL_EPOCHS):
                running_loss = 0.0
                final_model.train()
                for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_FINAL_EPOCHS}", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = final_model(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                train_loss = running_loss / len(train_loader)
                val_loss = validate_model(final_model, val_loader, criterion)
                
                print(f"Epoch {epoch+1}/{N_FINAL_EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = final_model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            
            print("Final model training complete.")
            
            print("\n--- 4. Evaluating Final Model on Test Set ---")
            final_model.load_state_dict(best_model_state)
            
            test_dataset = TabularDataset(X_test_tf, y_test)
            test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)
            
            final_model.eval()
            all_preds_scaled = []
            all_test_scaled = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    outputs = final_model(inputs)
                    all_preds_scaled.append(outputs.cpu().numpy())
                    all_test_scaled.append(targets.numpy())
            
            y_pred_scaled = np.concatenate(all_preds_scaled).flatten()
            y_test_scaled = np.concatenate(all_test_scaled).flatten()

            y_pred_final = 1.0 + y_pred_scaled
            y_test_final = 1.0 + y_test_scaled
            
            y_pred_clipped = y_pred_final.clip(1.0, 2.0)

            test_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_clipped))
            test_mae = mean_absolute_error(y_test_final, y_pred_clipped)
            test_r2 = r2_score(y_test_final, y_pred_clipped)

            print("\n--- Final PyTorch NN Model Test Results (v4 - Varol) ---")
            print(f"  Final Test RMSE: {test_rmse:.4f}")
            print(f"  Final Test MAE : {test_mae:.4f}")
            print(f"  Final Test R^2   : {test_r2:.4f}")

            print(f"\n--- 5. Saving Model ---")
            checkpoint = {
                "model_state_dict": best_model_state,
                "model_class": "MLP",
                "model_params": {
                    "n_features_in": n_features,
                    "n_layers": best_params['n_layers'],
                    "n_units": best_params['n_units'],
                    "dropout_rate": best_params['dropout_rate'],
                    "activation": best_params['activation']
                },
                "best_val_loss": best_val_loss,
                "optuna_best_params": best_params,
            }
            torch.save(checkpoint, MODEL_OUTPUT_FILE)
            print(f"Model checkpoint saved to {MODEL_OUTPUT_FILE}")
            
            print("\n✅ Process complete. Preprocessor (v4) and Model (v4) are saved.")