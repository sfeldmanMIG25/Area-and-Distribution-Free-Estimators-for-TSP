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

MODEL_DIR = os.path.join(ROOT_DIR, 'nn_model_pytorch_v6_twotower') # New folder
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'nn_alpha_model.pt')
CONT_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'nn_cont_preprocessor.joblib')
CAT_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'nn_cat_preprocessor.joblib')

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50       # Number of trials
EARLY_STOPPING_PATIENCE = 20 # More patience for a complex model
N_TRIAL_EPOCHS = 20        # Epochs for each *trial*
N_FINAL_EPOCHS = 200       # Max epochs for final training
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
    """
    Custom Dataset that handles two pre-transformed inputs:
    x_cont (continuous features) and x_cat (categorical features).
    """
    def __init__(self, X_cont, X_cat, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.float32)
        # Ensure y is 1D, then unsqueeze
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cont[idx], self.X_cat[idx], self.y[idx]

def build_preprocessors(X_train_df):
    """
    Builds TWO separate preprocessors: one for continuous data (using the
    "Varol" method) and one for categorical data.
    """
    print("Building SOTA 'Two-Tower' preprocessing pipelines...")
    
    # 1. Define feature groups
    categorical_features_list = ['dimension', 'grid_size']
    
    continuous_features_list = [
        col for col in X_train_df.columns if col not in categorical_features_list
    ]
    print(f"  Found {len(continuous_features_list)} continuous features.")
    print(f"  Found {len(categorical_features_list)} categorical features.")

    # 2. Create pipelines
    
    # Continuous "Varol" Pipeline:
    # Impute(0) + add_indicator (creates binary 'was_missing' features) + Scale
    continuous_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline:
    # Impute(frequent) + OneHotEncode
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Create and fit the two ColumnTransformers
    cont_preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipe, continuous_features_list)
        ],
        remainder='drop'
    )
    
    cat_preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_pipe, categorical_features_list)
        ],
        remainder='drop'
    )
    
    # Fit both preprocessors on the training data
    cont_preprocessor.fit(X_train_df)
    cat_preprocessor.fit(X_train_df)
    
    # 4. Get the output feature counts for the NN architecture
    n_cont_features = cont_preprocessor.transform(X_train_df).shape[1]
    n_cat_features = cat_preprocessor.transform(X_train_df).shape[1]
    
    print(f"  Continuous pipeline output features: {n_cont_features}")
    print(f"  Categorical pipeline output features: {n_cat_features}")
    
    return (
        cont_preprocessor, cat_preprocessor, 
        n_cont_features, n_cat_features,
        continuous_features_list, categorical_features_list
    )

def load_and_preprocess_main(data_path):
    """
    Main data loading function. Loads raw data and splits.
    """
    print("Loading raw data...")
    df = pd.read_csv(data_path)

    # Create scaled target [0, 1]
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha_scaled'] = df['alpha'].clip(1.0, 2.0) - 1.0
    
    y = df['alpha_scaled']

    # Define Features (X)
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'alpha_scaled', 'split', 'distribution_type'
    ]
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    
    # Critical fix for 'inf' values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Split Data (Train, Val, Test)
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

# --- 3. PYTORCH "TWO-TOWER" MODEL ---

def _build_mlp_tower(in_features, n_layers, n_units_list, activation, dropout_rate):
    """Helper function to build one tower with variable layer sizes."""
    layers = []
    for n_units in n_units_list:
        layers.append(nn.Linear(in_features, n_units))
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = n_units # Input for the next layer
    return nn.Sequential(*layers)

class TabularNet(nn.Module):
    """
    A "Two-Tower" SOTA model for tabular data.
    """
    def __init__(self, n_cont_features, n_cat_features, params):
        super(TabularNet, self).__init__()
        
        # Tower 1: Continuous Features
        self.cont_tower = _build_mlp_tower(
            in_features=n_cont_features,
            n_layers=params['n_layers_cont'],
            n_units_list=params['n_units_cont_list'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )
        
        # Tower 2: Categorical Features
        self.cat_tower = _build_mlp_tower(
            in_features=n_cat_features,
            n_layers=params['n_layers_cat'],
            n_units_list=params['n_units_cat_list'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )
        
        # Head: Combines the two tower outputs
        head_input_size = params['n_units_cont_list'][-1] + params['n_units_cat_list'][-1]
        
        self.head = _build_mlp_tower(
            in_features=head_input_size,
            n_layers=params['n_layers_head'],
            n_units_list=params['n_units_head_list'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(params['n_units_head_list'][-1], 1),
            nn.Sigmoid() # Constrain output to [0, 1]
        )

    def forward(self, x_cont, x_cat):
        """Forward pass takes two separate inputs."""
        cont_embedding = self.cont_tower(x_cont)
        cat_embedding = self.cat_tower(x_cat)
        
        # Concatenate the learned representations
        combined = torch.cat([cont_embedding, cat_embedding], dim=1)
        
        # Pass through the final head
        head_output = self.head(combined)
        return self.output_layer(head_output)

# --- 4. TRAINING & VALIDATION LOOPS ---

def train_one_epoch(model, loader, criterion, optimizer):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    for x_cont, x_cat, targets in loader:
        x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
        
        outputs = model(x_cont, x_cat)
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
        for x_cont, x_cat, targets in loader:
            x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
            outputs = model(x_cont, x_cat)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_loss = val_loss / len(loader)
    return avg_loss

# --- 5. OPTUNA HYPERPARAMETER TUNING ---

def objective(trial, X_train_cont_tf, X_train_cat_tf, y_train, X_val_cont_tf, X_val_cat_tf, y_val, n_cont_features, n_cat_features):
    """Optuna objective function for the Two-Tower model."""
    
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Continuous Tower
    n_layers_cont = trial.suggest_int('n_layers_cont', 2, 5)
    n_units_cont_list = []
    for i in range(n_layers_cont):
        n_units = trial.suggest_int(f'n_units_cont_{i}', 64, 256, log=True)
        n_units_cont_list.append(n_units)

    # Categorical Tower
    n_layers_cat = trial.suggest_int('n_layers_cat', 1, 3)
    n_units_cat_list = []
    for i in range(n_layers_cat):
        n_units = trial.suggest_int(f'n_units_cat_{i}', 16, 64, log=True)
        n_units_cat_list.append(n_units)

    # Head
    n_layers_head = trial.suggest_int('n_layers_head', 1, 3)
    n_units_head_list = []
    for i in range(n_layers_head):
        n_units = trial.suggest_int(f'n_units_head_{i}', 32, 128, log=True)
        n_units_head_list.append(n_units)

    params = {
        'n_layers_cont': n_layers_cont,
        'n_units_cont_list': n_units_cont_list,
        'n_layers_cat': n_layers_cat,
        'n_units_cat_list': n_units_cat_list,
        'n_layers_head': n_layers_head,
        'n_units_head_list': n_units_head_list,
        'activation': activation,
        'dropout_rate': dropout_rate,
    }
    
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train_cont_tf, X_train_cat_tf, y_train)
    val_dataset = TabularDataset(X_val_cont_tf, X_val_cat_tf, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model and optimizer
    model = TabularNet(n_cont_features, n_cat_features, params).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    best_val_loss = np.inf
    for epoch in range(N_TRIAL_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_model(model, val_loader, criterion)
        
        if np.isnan(val_loss) or np.isinf(val_loss):
            return np.inf # Return 'inf' to signal a failed trial
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
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

        # Build the two preprocessors
        (
            cont_preprocessor, cat_preprocessor, 
            n_cont_features, n_cat_features,
            cont_features_list, cat_features_list
        ) = build_preprocessors(X_train)
        
        print(f"Saving preprocessors to {MODEL_DIR}...")
        joblib.dump(cont_preprocessor, CONT_PREPROCESSOR_FILE)
        joblib.dump(cat_preprocessor, CAT_PREPROCESSOR_FILE)
        
        print("Transforming data splits...")
        X_train_cont_tf = cont_preprocessor.transform(X_train)
        X_train_cat_tf = cat_preprocessor.transform(X_train)
        
        X_val_cont_tf = cont_preprocessor.transform(X_val)
        X_val_cat_tf = cat_preprocessor.transform(X_val)
        
        X_test_cont_tf = cont_preprocessor.transform(X_test)
        X_test_cat_tf = cat_preprocessor.transform(X_test)
        
        X_train_full_cont_tf = cont_preprocessor.transform(X_train_full)
        X_train_full_cat_tf = cat_preprocessor.transform(X_train_full)
        
        print(f"Data transformed.")
        
        print(f"\n--- 2. Running Optuna Hyperparameter Tuning ({OPTUNA_N_TRIALS} trials) ---")
        objective_func = lambda trial: objective(
            trial, 
            X_train_cont_tf, X_train_cat_tf, y_train,
            X_val_cont_tf, X_val_cat_tf, y_val,
            n_cont_features, n_cat_features
        )
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
        
        if study.best_value == np.inf or np.isnan(study.best_value):
            print("\n\nCRITICAL ERROR: Optuna tuning failed. All trials resulted in 'inf' or 'nan'.")
        else:
            print(f"\nOptuna tuning complete. Best Validation MSE: {study.best_value:.6f}")
            print("Best parameters found:")
            print(study.best_params)

            print("\n--- 3. Training Final Model ---")
            best_params = study.best_params
            
            # Reconstruct the dynamic layer lists from the best params
            final_model_params = {
                'n_layers_cont': best_params['n_layers_cont'],
                'n_units_cont_list': [best_params[f'n_units_cont_{i}'] for i in range(best_params['n_layers_cont'])],
                'n_layers_cat': best_params['n_layers_cat'],
                'n_units_cat_list': [best_params[f'n_units_cat_{i}'] for i in range(best_params['n_layers_cat'])],
                'n_layers_head': best_params['n_layers_head'],
                'n_units_head_list': [best_params[f'n_units_head_{i}'] for i in range(best_params['n_layers_head'])],
                'activation': best_params['activation'],
                'dropout_rate': best_params['dropout_rate'],
            }
            
            # Create final datasets for training
            train_full_dataset = TabularDataset(X_train_full_cont_tf, X_train_full_cat_tf, y_train_full)
            val_size = int(len(train_full_dataset) * VALIDATION_SPLIT)
            train_size = len(train_full_dataset) - val_size
            train_data, val_data = random_split(train_full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_data, batch_size=best_params["batch_size"], shuffle=False)

            # Build final model
            final_model = TabularNet(n_cont_features, n_cat_features, final_model_params).to(device)
            
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
                for x_cont, x_cat, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_FINAL_EPOCHS}", leave=False):
                    x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
                    
                    outputs = final_model(x_cont, x_cat)
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
            
            test_dataset = TabularDataset(X_test_cont_tf, X_test_cat_tf, y_test)
            test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)
            
            final_model.eval()
            all_preds_scaled = []
            all_test_scaled = []
            with torch.no_grad():
                for x_cont, x_cat, targets in test_loader:
                    x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
                    outputs = final_model(x_cont, x_cat)
                    all_preds_scaled.append(outputs.cpu().numpy())
                    # --- THIS IS THE FIX ---
                    all_test_scaled.append(targets.cpu().numpy())
            
            y_pred_scaled = np.concatenate(all_preds_scaled).flatten()
            y_test_scaled = np.concatenate(all_test_scaled).flatten()

            y_pred_final = 1.0 + y_pred_scaled
            y_test_final = 1.0 + y_test_scaled
            
            y_pred_clipped = y_pred_final.clip(1.0, 2.0)

            test_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_clipped))
            test_mae = mean_absolute_error(y_test_final, y_pred_clipped)
            test_r2 = r2_score(y_test_final, y_pred_clipped)

            print("\n--- Final PyTorch 'Two-Tower' NN Model Test Results (v6) ---")
            print(f"  Final Test RMSE: {test_rmse:.4f}")
            print(f"  Final Test MAE : {test_mae:.4f}")
            print(f"  Final Test R^2   : {test_r2:.4f}")

            print(f"\n--- 5. Saving Model ---")
            checkpoint = {
                "model_state_dict": best_model_state,
                "model_class": "TabularNet",
                "model_params": {
                    "n_cont_features": n_cont_features,
                    "n_cat_features": n_cat_features,
                    **final_model_params # Add all tuned params
                },
                "best_val_loss": best_val_loss,
                "optuna_best_params": best_params,
                # Save the feature lists needed to run this model
                "cont_features_list": cont_features_list,
                "cat_features_list": cat_features_list
            }
            torch.save(checkpoint, MODEL_OUTPUT_FILE)
            print(f"Model checkpoint saved to {MODEL_OUTPUT_FILE}")
            
            print("\n✅ Process complete. Preprocessors (v6) and Model (v6) are saved.")