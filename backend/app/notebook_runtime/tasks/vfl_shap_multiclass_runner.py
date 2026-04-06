"""Standalone VFL SHAP multiclass (merged from VFL_SHAP_MultiClass.ipynb)."""
# VFL SHAP - Multi-Class Version
# All imports at the top
import os
import json
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import utility functions
from app.notebook_runtime.storage_paths import AGENTIC_FEATURES_JSON
from app.notebook_runtime.vfl_utils import (
    simplify_label,
    categorize_feature_by_evidence,
    format_action_readable,
    ATTACK_ACTIONS,
    load_agent_definitions,
    split_features_by_agent_definitions,
    get_evidence_type,
    get_agent_actions_for_attack,
)

# Import model classes
from app.notebook_runtime.model_utils import (
    LocalEncoder,
    ActiveClassifier,
    VFLModel,
    AgentMetaModel,
    StandardNeuralNetwork,
)

# -----------------------------


# 1. Load Dataset & Extract Multi-Class Labels
# -----------------------------
# Dataset folder path
DATASETS_FOLDER = Path("datasets")

# Find all CSV files in the datasets folder (only standard .csv files)
csv_files = [f for f in DATASETS_FOLDER.glob("*.csv") if f.suffix == ".csv"]

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {DATASETS_FOLDER}")

print(f"Found {len(csv_files)} CSV file(s) in {DATASETS_FOLDER}:")
for csv_file in csv_files:
    print(f"  - {csv_file.name}")

# Load all CSV files and concatenate them
print(f"\nLoading and concatenating CSV files...")
dataframes = []
for csv_file in csv_files:
    print(f"  Loading {csv_file.name}...")
    df_temp = pd.read_csv(csv_file)
    print(f"    Rows: {len(df_temp)}, Columns: {len(df_temp.columns)}")
    dataframes.append(df_temp)

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal combined rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Drop unnecessary columns
df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"], errors="ignore")
print(f"After dropping columns: {len(df.columns)} columns remaining")


# Extract and simplify label information
label_col = "label"

# Group similar attack types
# Apply simplification
df['label_simplified'] = df[label_col].apply(simplify_label)

# Group classes with less than 200 rows into "OTHERS"
label_counts = df['label_simplified'].value_counts()
min_rows = 200

# Find labels with less than min_rows
small_labels = label_counts[label_counts < min_rows].index.tolist()

if len(small_labels) > 0:
    print(f"Grouping {len(small_labels)} classes with < {min_rows} rows into 'OTHERS':")
    for small_label in small_labels:
        count = label_counts[small_label]
        print(f"  {small_label}: {count} rows -> OTHERS")
    
    # Replace small labels with "OTHERS"
    df.loc[df['label_simplified'].isin(small_labels), 'label_simplified'] = 'OTHERS'

# Create mapping to numeric
unique_labels = sorted(df['label_simplified'].unique())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
label_names_short = {idx: label for label, idx in label_mapping.items()}

df['label_numeric'] = df['label_simplified'].map(label_mapping)
label_col = 'label_numeric'
num_classes = len(unique_labels)

print(f"Found {num_classes} simplified label types:")
for orig_label in sorted(df['label_simplified'].unique()):
    num = label_mapping[orig_label]
    print(f"  {orig_label} -> {num}")

label_mapping_dict = label_names_short

print(f"\nNumber of classes: {num_classes}")
print(f"Label distribution:\n{df[label_col].value_counts().sort_index()}")
# -----------------------------

# 2. Agent-Based Vertical Feature Partition (from agentic_features.json)
# -----------------------------
# Features are split by RAN / Edge / Core agents; names and actions from JSON.
# vfl_utils imports are in Cell 0

non_feature_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "label", "label_numeric", "label_simplified"]
all_features = [col for col in df.columns if col not in non_feature_cols]

# Filter to only numeric columns (exclude string/object columns)
all_features = [col for col in all_features if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

print(f"Total features: {len(all_features)}")

# Load agent definitions (RAN, Edge, Core) and split features by JSON (backend/storage)
agent_definitions = load_agent_definitions(AGENTIC_FEATURES_JSON)
agent1_features, agent2_features, agent3_features, feature_categories = split_features_by_agent_definitions(
    all_features, agent_definitions
)

print(f"\nAgent 1 (RAN): {len(agent1_features)} features")
print(f"Agent 2 (Edge): {len(agent2_features)} features")
print(f"Agent 3 (Core): {len(agent3_features)} features")

agent_sizes = [len(agent1_features), len(agent2_features), len(agent3_features)]
total_features = sum(agent_sizes)
target_size = total_features / 3
balance_ratio = (max(agent_sizes) - min(agent_sizes)) / target_size if target_size > 0 else 0
print(f"\nBalance check:")
print(f"  Target size per agent: {target_size:.1f}")
print(f"  Actual sizes: {agent_sizes}")
print(f"  Balance ratio: {balance_ratio:.2%} (lower is better)")

# Agent names, domains, and actions from agentic definitions (for display)
agent_names = agent_definitions["agent_names"]
agent_domains = agent_definitions["agent_domains"]
# Format agent_actions as one string per agent (list of action strings -> single string)
agent_actions = [
    ", ".join(agent_definitions["agent_actions"][i]) if agent_definitions["agent_actions"][i] else ""
    for i in range(3)
]

agent_feature_groups = [
    f"Features: {', '.join(agent1_features[:3])}..." if len(agent1_features) > 3 else f"Features: {', '.join(agent1_features)}",
    f"Features: {', '.join(agent2_features[:3])}..." if len(agent2_features) > 3 else f"Features: {', '.join(agent2_features)}",
    f"Features: {', '.join(agent3_features[:3])}..." if len(agent3_features) > 3 else f"Features: {', '.join(agent3_features)}"
]

# Per-attack suggested actions for SHAP (from ATTACK_ACTIONS / get_agent_actions_for_attack; not from attack_options)
agent_action_mapping = {}
agent_feature_lists = [agent1_features, agent2_features, agent3_features]
for i in range(3):
    agent_action_mapping[i] = {}
    for attack_type in ATTACK_ACTIONS.keys():
        agent_action_mapping[i][attack_type] = get_agent_actions_for_attack(agent_feature_lists[i], attack_type)

print("\n=== Agent Configuration ===")
for i in range(3):
    features_list = [agent1_features, agent2_features, agent3_features][i]
    print(f"\n{agent_names[i]}:")
    print(f"  Domain: {agent_domains[i]}")
    print(f"  Features ({len(features_list)}): {features_list}")

# -----------------------------

# 3. Preprocess Features
# -----------------------------
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()

X1 = torch.tensor(scaler1.fit_transform(df[agent1_features].values), dtype=torch.float32)
X2 = torch.tensor(scaler2.fit_transform(df[agent2_features].values), dtype=torch.float32)
X3 = torch.tensor(scaler3.fit_transform(df[agent3_features].values), dtype=torch.float32)

# Ensure we use numeric labels
if 'label_numeric' in df.columns:
    y = torch.tensor(df['label_numeric'].values, dtype=torch.long)  # long for multi-class
else:
    # Fallback: convert label_col to numeric if needed
    if df[label_col].dtype == 'object' or df[label_col].dtype == 'string':
        # Convert string labels to numeric
        unique_labels = sorted(df[label_col].unique())
        label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
        y = torch.tensor(df[label_col].map(label_to_num).values, dtype=torch.long)
    else:
        y = torch.tensor(df[label_col].values, dtype=torch.long)

x_parts = [X1, X2, X3]
# -----------------------------

# 4. Train/Val/Test Split (Improved Stratification)
# -----------------------------
# Use numeric labels for stratification
stratify_labels = df['label_numeric'] if 'label_numeric' in df.columns else y.numpy()

# Check class distribution before split
print("Class distribution before split:")
print(stratify_labels.value_counts().sort_index())

# First split: train+val (80%) vs test (20%)
trainval_idx, test_idx = train_test_split(
    range(len(df)),
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels
)

# Second split: train (64%) vs val (16%) from train+val
trainval_labels = stratify_labels.iloc[trainval_idx] if isinstance(stratify_labels, pd.Series) else stratify_labels[trainval_idx]
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.2,  # 20% of 80% = 16% of total
    random_state=42,
    stratify=trainval_labels
)

# Create splits
x_train_parts = [x[train_idx] for x in x_parts]
x_val_parts = [x[val_idx] for x in x_parts]
x_test_parts = [x[test_idx] for x in x_parts]
y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

# Verify stratification
print("\nTrain set class distribution:")
train_labels = stratify_labels.iloc[train_idx] if isinstance(stratify_labels, pd.Series) else stratify_labels[train_idx]
print(pd.Series(train_labels).value_counts().sort_index())

print("\nValidation set class distribution:")
val_labels = stratify_labels.iloc[val_idx] if isinstance(stratify_labels, pd.Series) else stratify_labels[val_idx]
print(pd.Series(val_labels).value_counts().sort_index())

print("\nTest set class distribution:")
test_labels = stratify_labels.iloc[test_idx] if isinstance(stratify_labels, pd.Series) else stratify_labels[test_idx]
print(pd.Series(test_labels).value_counts().sort_index())

print(f"\nSplit sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
# -----------------------------

# 5. Multi-Class VFL Model (Improved Architecture)
# -----------------------------
# Model classes are now imported from model_utils.py
# LocalEncoder, ActiveClassifier, VFLModel, and AgentMetaModel are available
# -----------------------------

# 6. Train & Evaluate Functions (Multi-Class with Validation & Early Stopping)
# -----------------------------
def evaluate_model(model, x_parts, y, criterion=None):
    """Evaluate model and return loss, accuracy, recall, f1"""
    model.eval()
    with torch.no_grad():
        y_hat = model(x_parts)
        y_pred = torch.argmax(y_hat, dim=1)
        
        loss = None
        if criterion is not None:
            loss = criterion(y_hat, y.long()).item()
        
        acc = accuracy_score(y.cpu(), y_pred.cpu())
        rec = recall_score(y.cpu(), y_pred.cpu(), average='macro', zero_division=0)
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='macro', zero_division=0)
        
    return loss, acc, rec, f1

def train_vfl(model, x_train_parts, y_train, x_val_parts, y_val, 
              epochs=100, lr=1e-3, use_class_weights=True, 
              early_stop_patience=20, min_delta=0.001, eval_every=5):
    """
    Train VFL model with validation, early stopping, and best model checkpointing
    
    Args:
        model: VFL model to train
        x_train_parts: Training features for each agent
        y_train: Training labels
        x_val_parts: Validation features for each agent
        y_val: Validation labels
        epochs: Maximum epochs
        lr: Learning rate
        use_class_weights: Use class weights for imbalanced data
        early_stop_patience: Early stopping patience (epochs)
        min_delta: Minimum improvement to reset patience
        eval_every: Evaluate validation every N epochs
    """
    # Calculate class weights for imbalanced data (clamped to prevent extreme values)
    if use_class_weights:
        y_np = y_train.cpu().numpy()
        class_counts = np.bincount(y_np, minlength=model.num_classes)
        total = len(y_np)
        class_weights = total / (len(class_counts) * class_counts.astype(float))
        # Clamp weights to prevent extreme values (max weight = 20)
        class_weights = np.clip(class_weights, None, 20.0)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights (clamped max=20): {class_weights.numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Scheduler uses validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, 
        threshold=0.0001, min_lr=1e-6
    )
    
    # Early stopping tracking
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training with early stopping (patience={early_stop_patience}, min_delta={min_delta})")
    print(f"Evaluating validation every {eval_every} epochs\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        y_hat_train = model(x_train_parts)
        train_loss = criterion(y_hat_train, y_train.long())
        
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation phase (every eval_every epochs or at start/end)
        if epoch % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
            val_loss, val_acc, val_rec, val_f1 = evaluate_model(model, x_val_parts, y_val, criterion)
            
            # Update scheduler with validation loss (fix warning by using .item())
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Check for improvement
            improved = False
            if val_f1 > best_val_f1 + min_delta:
                improved = True
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
                print(f"[VFL] Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
                      f"Val Rec: {val_rec:.4f} | LR: {current_lr:.6f} | ✓ BEST")
            else:
                patience_counter += 1
                print(f"[VFL] Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
                      f"Val Rec: {val_rec:.4f} | LR: {current_lr:.6f} | "
                      f"Patience: {patience_counter}/{early_stop_patience}")
            
            # Early stopping check
            if patience_counter >= early_stop_patience:
                print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs. "
                      f"Best F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break
        
        # Print training progress (every 10 epochs if not evaluating)
        elif epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[VFL] Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | LR: {current_lr:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[CHECKPOINT] Loaded best model from epoch {best_epoch} (Val F1: {best_val_f1:.4f})")
    
    return best_epoch, best_val_f1, best_val_loss

def evaluate(model, x_parts, y, print_per_class=False):
    """Comprehensive evaluation with per-class metrics"""
    model.eval()
    with torch.no_grad():
        y_hat = model(x_parts)
        y_pred = torch.argmax(y_hat, dim=1)
    
    acc = accuracy_score(y.cpu(), y_pred.cpu())
    rec_macro = recall_score(y.cpu(), y_pred.cpu(), average='macro', zero_division=0)
    rec_weighted = recall_score(y.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
    f1_macro = f1_score(y.cpu(), y_pred.cpu(), average='macro', zero_division=0)
    f1_weighted = f1_score(y.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {rec_macro:.4f} | Weighted Recall: {rec_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")
    
    if print_per_class:
        print(f"\n=== Per-Class Metrics ===")
        class_names = [label_mapping_dict.get(i, f"Class_{i}") for i in range(model.num_classes)]
        print(classification_report(y.cpu(), y_pred.cpu(), target_names=class_names, zero_division=0))
    
    return acc, rec_macro, f1_macro
# -----------------------------

# 7. Train VFL Model with Validation & Early Stopping
# -----------------------------
embed_dim = 64
hidden_dim = 128
model = VFLModel(
    input_dims=[len(agent1_features), len(agent2_features), len(agent3_features)],
    embed_dim=embed_dim,
    num_classes=num_classes,
    hidden_dim=hidden_dim
)

print(f"Model parameters: embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")
print(f"Fusion: concat (embed_dim*3 = {embed_dim*3})")
print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Define training parameters (will be reused for Standard NN)
training_epochs = 100
training_lr = 0.001
training_use_class_weights = True
training_early_stop_patience = 20
training_min_delta = 0.001
training_eval_every = 5

# Train with validation and early stopping
best_epoch, best_val_f1, best_val_loss = train_vfl(
    model, x_train_parts, y_train, x_val_parts, y_val,
    epochs=training_epochs, lr=training_lr, use_class_weights=training_use_class_weights,
    early_stop_patience=training_early_stop_patience, min_delta=training_min_delta, eval_every=training_eval_every
)

# Final evaluation on test set
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)
acc, rec, f1 = evaluate(model, x_test_parts, y_test, print_per_class=True)

# Confusion matrix
y_test_pred = torch.argmax(model(x_test_parts), dim=1).cpu().numpy()
y_test_np = y_test.cpu().numpy()
cm = confusion_matrix(y_test_np, y_test_pred)
print(f"\n=== Confusion Matrix ===")
print("Rows = True labels, Columns = Predicted labels")
class_names = [label_mapping_dict.get(i, f"Class_{i}") for i in range(num_classes)]
print("Classes:", class_names)
print(cm)

# Save VFL performance metrics
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
vfl_perf_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro_Recall", "Macro_F1", "Best_Val_F1", "Best_Epoch"],
    "Value": [acc, rec, f1, best_val_f1, best_epoch]
})
vfl_perf_filename = f"outputs/vfl_shap_performance_{timestamp}.csv"
vfl_perf_df.to_csv(vfl_perf_filename, index=False)
print(f"\nVFL Performance saved to {vfl_perf_filename}")

# Store VFL results for comparison
vfl_results = {
    "model_type": "VFL",
    "accuracy": float(acc),
    "macro_recall": float(rec),
    "macro_f1": float(f1),
    "best_val_f1": float(best_val_f1),
    "best_epoch": int(best_epoch),
    "best_val_loss": float(best_val_loss),
    "num_classes": num_classes,
    "embed_dim": embed_dim,
    "hidden_dim": hidden_dim,
    "num_agents": 3,
    "train_size": len(y_train),
    "val_size": len(y_val),
    "test_size": len(y_test),
    "training_params": {
        "epochs": training_epochs,
        "learning_rate": training_lr,
        "use_class_weights": training_use_class_weights,
        "early_stop_patience": training_early_stop_patience,
        "min_delta": training_min_delta,
        "eval_every": training_eval_every
    }
}
# -----------------------------

# 7.5. Train Standard (Non-Federated) Neural Network for Comparison
# -----------------------------
# StandardNeuralNetwork is already imported at the top of the notebook

print("="*80)
print("TRAINING STANDARD (NON-FEDERATED) NEURAL NETWORK")
print("="*80)

# Concatenate all features for standard neural network (non-federated)
# This uses all features together, not split by agent
all_features_list = agent1_features + agent2_features + agent3_features
X_train_standard = torch.cat([X1[train_idx], X2[train_idx], X3[train_idx]], dim=1)
X_val_standard = torch.cat([X1[val_idx], X2[val_idx], X3[val_idx]], dim=1)
X_test_standard = torch.cat([X1[test_idx], X2[test_idx], X3[test_idx]], dim=1)

print(f"\nStandard NN input dimension: {X_train_standard.shape[1]}")
print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Initialize standard neural network
# Use similar architecture depth to VFL for fair comparison
standard_model = StandardNeuralNetwork(
    input_dim=X_train_standard.shape[1],
    num_classes=num_classes,
    hidden_dims=[256, 128, 64],  # Similar capacity to VFL
    dropout=0.2
)

print(f"Standard NN parameters: input_dim={X_train_standard.shape[1]}, num_classes={num_classes}")

# Train standard model with same settings as VFL
def train_standard_nn(model, x_train, y_train, x_val, y_val, 
                     epochs=100, lr=1e-3, use_class_weights=True,
                     early_stop_patience=20, min_delta=0.001, eval_every=5):
    """Train standard neural network with same training setup as VFL"""
    # Calculate class weights (same as VFL)
    if use_class_weights:
        y_np = y_train.cpu().numpy()
        class_counts = np.bincount(y_np, minlength=model.num_classes)
        total = len(y_np)
        class_weights = total / (len(class_counts) * class_counts.astype(float))
        class_weights = np.clip(class_weights, None, 20.0)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights (clamped max=20): {class_weights.numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15,
        threshold=0.0001, min_lr=1e-6
    )
    
    # Early stopping tracking
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training with early stopping (patience={early_stop_patience}, min_delta={min_delta})")
    print(f"Evaluating validation every {eval_every} epochs\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        y_hat_train = model(x_train)
        train_loss = criterion(y_hat_train, y_train.long())
        
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation phase
        if epoch % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
            # Evaluate standard model directly
            model.eval()
            with torch.no_grad():
                y_hat_val = model(x_val)
                y_pred_val = torch.argmax(y_hat_val, dim=1)
                val_loss = criterion(y_hat_val, y_val.long()).item()
                val_acc = accuracy_score(y_val.cpu(), y_pred_val.cpu())
                val_rec = recall_score(y_val.cpu(), y_pred_val.cpu(), average='macro', zero_division=0)
                val_f1 = f1_score(y_val.cpu(), y_pred_val.cpu(), average='macro', zero_division=0)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Check for improvement
            improved = False
            if val_f1 > best_val_f1 + min_delta:
                improved = True
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"[STD] Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
                      f"Val Rec: {val_rec:.4f} | LR: {current_lr:.6f} | ✓ BEST")
            else:
                patience_counter += 1
                print(f"[STD] Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
                      f"Val Rec: {val_rec:.4f} | LR: {current_lr:.6f} | "
                      f"Patience: {patience_counter}/{early_stop_patience}")
            
            # Early stopping check
            if patience_counter >= early_stop_patience:
                print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs. "
                      f"Best F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break
        
        # Print training progress
        elif epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[STD] Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | LR: {current_lr:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[CHECKPOINT] Loaded best model from epoch {best_epoch} (Val F1: {best_val_f1:.4f})")
    
    return best_epoch, best_val_f1, best_val_loss

# Train standard model with SAME parameters as VFL
print(f"\nUsing same training parameters as VFL:")
print(f"  Epochs: {training_epochs}")
print(f"  Learning Rate: {training_lr}")
print(f"  Class Weights: {training_use_class_weights}")
print(f"  Early Stop Patience: {training_early_stop_patience}")
print(f"  Min Delta: {training_min_delta}")
print(f"  Eval Every: {training_eval_every}")

standard_best_epoch, standard_best_val_f1, standard_best_val_loss = train_standard_nn(
    standard_model, X_train_standard, y_train, X_val_standard, y_val,
    epochs=training_epochs, lr=training_lr, use_class_weights=training_use_class_weights,
    early_stop_patience=training_early_stop_patience, min_delta=training_min_delta, eval_every=training_eval_every
)

# Final evaluation on test set
print("\n" + "="*60)
print("STANDARD NN FINAL TEST SET EVALUATION")
print("="*60)

# Evaluate standard model (need to call model directly, not through evaluate function)
standard_model.eval()
with torch.no_grad():
    y_hat_standard = standard_model(X_test_standard)
    y_pred_standard = torch.argmax(y_hat_standard, dim=1)

standard_acc = accuracy_score(y_test.cpu(), y_pred_standard.cpu())
standard_rec = recall_score(y_test.cpu(), y_pred_standard.cpu(), average='macro', zero_division=0)
standard_f1 = f1_score(y_test.cpu(), y_pred_standard.cpu(), average='macro', zero_division=0)
standard_rec_weighted = recall_score(y_test.cpu(), y_pred_standard.cpu(), average='weighted', zero_division=0)
standard_f1_weighted = f1_score(y_test.cpu(), y_pred_standard.cpu(), average='weighted', zero_division=0)

print(f"\n=== Evaluation Results ===")
print(f"Accuracy: {standard_acc:.4f}")
print(f"Macro Recall: {standard_rec:.4f} | Weighted Recall: {standard_rec_weighted:.4f}")
print(f"Macro F1: {standard_f1:.4f} | Weighted F1: {standard_f1_weighted:.4f}")

print(f"\n=== Per-Class Metrics ===")
class_names = [label_mapping_dict.get(i, f"Class_{i}") for i in range(num_classes)]
print(classification_report(y_test.cpu(), y_pred_standard.cpu(), target_names=class_names, zero_division=0))

# Confusion matrix for standard model
y_test_pred_standard = torch.argmax(standard_model(X_test_standard), dim=1).cpu().numpy()
y_test_np = y_test.cpu().numpy()
cm_standard = confusion_matrix(y_test_np, y_test_pred_standard)
print(f"\n=== Standard NN Confusion Matrix ===")
print("Rows = True labels, Columns = Predicted labels")
class_names = [label_mapping_dict.get(i, f"Class_{i}") for i in range(num_classes)]
print("Classes:", class_names)
print(cm_standard)

# Store standard NN results for comparison
standard_results = {
    "model_type": "Standard_NN",
    "accuracy": float(standard_acc),
    "macro_recall": float(standard_rec),
    "macro_f1": float(standard_f1),
    "best_val_f1": float(standard_best_val_f1),
    "best_epoch": int(standard_best_epoch),
    "best_val_loss": float(standard_best_val_loss),
    "num_classes": num_classes,
    "input_dim": X_train_standard.shape[1],
    "hidden_dims": [256, 128, 64],
    "train_size": len(y_train),
    "val_size": len(y_val),
    "test_size": len(y_test),
    "training_params": {
        "epochs": training_epochs,
        "learning_rate": training_lr,
        "use_class_weights": training_use_class_weights,
        "early_stop_patience": training_early_stop_patience,
        "min_delta": training_min_delta,
        "eval_every": training_eval_every
    }
}

print(f"\nStandard NN Performance saved to comparison file")
# -----------------------------

# 7.6. Model Comparison and Results Output
# -----------------------------
print("\n" + "="*80)
print("MODEL COMPARISON AND RESULTS ANALYSIS")
print("="*80)

# Create comprehensive comparison output
comparison_data = {
    "experiment_info": {
        "timestamp": timestamp,
        "dataset_info": {
            "total_features": len(all_features_list),
            "num_classes": num_classes,
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "class_names": [label_mapping_dict.get(i, f"Class_{i}") for i in range(num_classes)]
        }
    },
    "vfl_model": vfl_results,
    "standard_nn_model": standard_results,
    "comparison": {
        "accuracy_diff": float(standard_acc - acc),
        "recall_diff": float(standard_rec - rec),
        "f1_diff": float(standard_f1 - f1),
        "val_f1_diff": float(standard_best_val_f1 - best_val_f1),
        "best_accuracy": "VFL" if acc > standard_acc else "Standard_NN",
        "best_f1": "VFL" if f1 > standard_f1 else "Standard_NN",
        "best_recall": "VFL" if rec > standard_rec else "Standard_NN"
    }
}

# Save comparison as JSON
comparison_json_path = f"outputs/model_comparison_{timestamp}.json"
with open(comparison_json_path, 'w', encoding='utf-8') as f:
    json.dump(comparison_data, f, indent=2, ensure_ascii=False)
print(f"✓ Comparison JSON saved to {comparison_json_path}")

# Create CSV comparison table
comparison_df = pd.DataFrame([
    {
        "Model": "VFL",
        "Accuracy": vfl_results["accuracy"],
        "Macro_Recall": vfl_results["macro_recall"],
        "Macro_F1": vfl_results["macro_f1"],
        "Best_Val_F1": vfl_results["best_val_f1"],
        "Best_Epoch": vfl_results["best_epoch"],
        "Best_Val_Loss": vfl_results["best_val_loss"],
        "Architecture": f"VFL (3 agents, embed_dim={vfl_results['embed_dim']}, hidden_dim={vfl_results['hidden_dim']})"
    },
    {
        "Model": "Standard_NN",
        "Accuracy": standard_results["accuracy"],
        "Macro_Recall": standard_results["macro_recall"],
        "Macro_F1": standard_results["macro_f1"],
        "Best_Val_F1": standard_results["best_val_f1"],
        "Best_Epoch": standard_results["best_epoch"],
        "Best_Val_Loss": standard_results["best_val_loss"],
        "Architecture": f"Standard NN (hidden_dims={standard_results['hidden_dims']})"
    }
])

comparison_csv_path = f"outputs/model_comparison_{timestamp}.csv"
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"✓ Comparison CSV saved to {comparison_csv_path}")

# Create detailed text report
report_lines = []
report_lines.append("="*80)
report_lines.append("MODEL COMPARISON REPORT")
report_lines.append("="*80)
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

# Dataset info
report_lines.append("DATASET INFORMATION")
report_lines.append("-"*80)
report_lines.append(f"Total Features: {len(all_features_list)}")
report_lines.append(f"Number of Classes: {num_classes}")
report_lines.append(f"Class Names: {', '.join([label_mapping_dict.get(i, f'Class_{i}') for i in range(num_classes)])}")
report_lines.append(f"Train Size: {len(y_train)}")
report_lines.append(f"Validation Size: {len(y_val)}")
report_lines.append(f"Test Size: {len(y_test)}")
report_lines.append("")

# VFL Model Results
report_lines.append("="*80)
report_lines.append("VFL MODEL RESULTS")
report_lines.append("="*80)
report_lines.append(f"Architecture: Vertical Federated Learning (3 agents)")
report_lines.append(f"  - Embedding Dimension: {vfl_results['embed_dim']}")
report_lines.append(f"  - Hidden Dimension: {vfl_results['hidden_dim']}")
report_lines.append(f"  - Number of Agents: {vfl_results['num_agents']}")
report_lines.append(f"  - Agent 1 Features: {len(agent1_features)}")
report_lines.append(f"  - Agent 2 Features: {len(agent2_features)}")
report_lines.append(f"  - Agent 3 Features: {len(agent3_features)}")
report_lines.append("")
report_lines.append("Performance Metrics:")
report_lines.append(f"  - Test Accuracy: {vfl_results['accuracy']:.4f}")
report_lines.append(f"  - Macro Recall: {vfl_results['macro_recall']:.4f}")
report_lines.append(f"  - Macro F1: {vfl_results['macro_f1']:.4f}")
report_lines.append(f"  - Best Validation F1: {vfl_results['best_val_f1']:.4f}")
report_lines.append(f"  - Best Validation Loss: {vfl_results['best_val_loss']:.4f}")
report_lines.append(f"  - Best Epoch: {vfl_results['best_epoch']}")
report_lines.append("")
report_lines.append("Training Parameters:")
report_lines.append(f"  - Epochs: {vfl_results['training_params']['epochs']}")
report_lines.append(f"  - Learning Rate: {vfl_results['training_params']['learning_rate']}")
report_lines.append(f"  - Class Weights: {vfl_results['training_params']['use_class_weights']}")
report_lines.append(f"  - Early Stop Patience: {vfl_results['training_params']['early_stop_patience']}")
report_lines.append("")

# Standard NN Results
report_lines.append("="*80)
report_lines.append("STANDARD NEURAL NETWORK RESULTS")
report_lines.append("="*80)
report_lines.append(f"Architecture: Standard (Non-Federated) Neural Network")
report_lines.append(f"  - Input Dimension: {standard_results['input_dim']}")
report_lines.append(f"  - Hidden Dimensions: {standard_results['hidden_dims']}")
report_lines.append("")
report_lines.append("Performance Metrics:")
report_lines.append(f"  - Test Accuracy: {standard_results['accuracy']:.4f}")
report_lines.append(f"  - Macro Recall: {standard_results['macro_recall']:.4f}")
report_lines.append(f"  - Macro F1: {standard_results['macro_f1']:.4f}")
report_lines.append(f"  - Best Validation F1: {standard_results['best_val_f1']:.4f}")
report_lines.append(f"  - Best Validation Loss: {standard_results['best_val_loss']:.4f}")
report_lines.append(f"  - Best Epoch: {standard_results['best_epoch']}")
report_lines.append("")
report_lines.append("Training Parameters:")
report_lines.append(f"  - Epochs: {standard_results['training_params']['epochs']}")
report_lines.append(f"  - Learning Rate: {standard_results['training_params']['learning_rate']}")
report_lines.append(f"  - Class Weights: {standard_results['training_params']['use_class_weights']}")
report_lines.append(f"  - Early Stop Patience: {standard_results['training_params']['early_stop_patience']}")
report_lines.append("")
report_lines.append("Note: Standard NN uses IDENTICAL training parameters as VFL for fair comparison")
report_lines.append("")

# Comparison Summary
report_lines.append("="*80)
report_lines.append("COMPARISON SUMMARY")
report_lines.append("="*80)
report_lines.append(f"Accuracy Difference (Standard_NN - VFL): {comparison_data['comparison']['accuracy_diff']:+.4f}")
report_lines.append(f"Recall Difference (Standard_NN - VFL): {comparison_data['comparison']['recall_diff']:+.4f}")
report_lines.append(f"F1 Difference (Standard_NN - VFL): {comparison_data['comparison']['f1_diff']:+.4f}")
report_lines.append(f"Validation F1 Difference (Standard_NN - VFL): {comparison_data['comparison']['val_f1_diff']:+.4f}")
report_lines.append("")
report_lines.append("Best Performing Model:")
report_lines.append(f"  - Best Accuracy: {comparison_data['comparison']['best_accuracy']}")
report_lines.append(f"  - Best F1 Score: {comparison_data['comparison']['best_f1']}")
report_lines.append(f"  - Best Recall: {comparison_data['comparison']['best_recall']}")
report_lines.append("")
report_lines.append("="*80)

# Save text report
report_text_path = f"outputs/model_comparison_report_{timestamp}.txt"
with open(report_text_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"✓ Comparison report saved to {report_text_path}")

# Print summary to console
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"Accuracy: VFL={vfl_results['accuracy']:.4f}, Standard_NN={standard_results['accuracy']:.4f} (Diff: {comparison_data['comparison']['accuracy_diff']:+.4f})")
print(f"F1 Score: VFL={vfl_results['macro_f1']:.4f}, Standard_NN={standard_results['macro_f1']:.4f} (Diff: {comparison_data['comparison']['f1_diff']:+.4f})")
print(f"Recall: VFL={vfl_results['macro_recall']:.4f}, Standard_NN={standard_results['macro_recall']:.4f} (Diff: {comparison_data['comparison']['recall_diff']:+.4f})")
print(f"\nBest Model:")
print(f"  Accuracy: {comparison_data['comparison']['best_accuracy']}")
print(f"  F1 Score: {comparison_data['comparison']['best_f1']}")
print(f"  Recall: {comparison_data['comparison']['best_recall']}")
print("="*80)

print(f"\n✓ All comparison files saved with timestamp: {timestamp}")
# -----------------------------

# 8. Build Agent-Level Meta-Features (Full Embeddings)
# -----------------------------
# Use FULL agent embeddings instead of mean to preserve information
# This improves meta-model learning significantly
model.eval()
with torch.no_grad():
    train_embeds = model.get_agent_embeddings(x_train_parts)  # list of (N, embed_dim=64)
    test_embeds = model.get_agent_embeddings(x_test_parts)     # list of (N, embed_dim=64)

# Concatenate full embeddings: (N, 64) + (N, 64) + (N, 64) = (N, 192)
# This preserves all information instead of collapsing to 3 scalars
X_train_meta = torch.cat([train_embeds[0], train_embeds[1], train_embeds[2]], dim=1)  # (N, 192)
X_test_meta = torch.cat([test_embeds[0], test_embeds[1], test_embeds[2]], dim=1)        # (N, 192)

print(f"Meta-features shape: Train={X_train_meta.shape}, Test={X_test_meta.shape}")
print(f"Using full embeddings (192 dims) instead of mean (3 dims) for better meta-model learning")
# -----------------------------

# 9. Multi-Class Meta-Model (Improved: MLP + Soft Distillation)
# -----------------------------
# Upgraded meta-model: MLP architecture + soft distillation with temperature
# This significantly improves agreement between VFL model and meta-model
# AgentMetaModel is now imported from model_utils.py

# Initialize meta-model with full embedding dimension (192)
meta_model = AgentMetaModel(in_dim=X_train_meta.shape[1], num_classes=num_classes, hidden_dim=128)
print(f"Meta-model input dim: {X_train_meta.shape[1]}, output classes: {num_classes}")

# Soft distillation: KL divergence with temperature (preserves teacher confidence)
T = 3.0  # Temperature for softmax smoothing
kl_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(meta_model.parameters(), lr=1e-3, weight_decay=1e-4)
meta_epochs = 50

print(f"\nTraining meta-model with soft distillation (T={T})...")
model.eval()
for epoch in range(meta_epochs):
    meta_model.train()
    optimizer.zero_grad()
    
    # Teacher: get logits from VFL model
    with torch.no_grad():
        teacher_logits = model(x_train_parts)  # (N, num_classes) logits
        # Soft targets: apply temperature to teacher logits
        teacher_probs_T = torch.softmax(teacher_logits / T, dim=1)
    
    # Student: get logits from meta-model
    student_logits = meta_model(X_train_meta)  # (N, num_classes) logits
    # Soft predictions: apply temperature and log
    student_log_probs_T = torch.log_softmax(student_logits / T, dim=1)
    
    # KL divergence loss (scaled by T^2 for proper gradient scaling)
    loss = kl_loss(student_log_probs_T, teacher_probs_T) * (T * T)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"[META] Epoch {epoch} | Distillation Loss: {loss.item():.4f}")

# Meta-model fidelity evaluation
print(f"\n=== Meta-Model Fidelity Evaluation ===")
meta_model.eval()
with torch.no_grad():
    # Get predictions from both models
    teacher_test = model(x_test_parts)              # (N, num_classes) logits
    student_test = meta_model(X_test_meta)         # (N, num_classes) logits
    
    # Hard predictions (argmax)
    teacher_pred = teacher_test.argmax(dim=1).cpu().numpy()
    student_pred = student_test.argmax(dim=1).cpu().numpy()
    
    # Agreement metric (accuracy between teacher and student predictions)
    meta_acc = accuracy_score(teacher_pred, student_pred)
    
    # Additional metrics: KL divergence and logit MSE
    teacher_probs_test = torch.softmax(teacher_test, dim=1)
    student_probs_test = torch.softmax(student_test, dim=1)
    kl_div = kl_loss(torch.log(student_probs_test + 1e-8), teacher_probs_test).item()
    logit_mse = torch.mean((teacher_test - student_test) ** 2).item()

print(f"[META] Agreement (Meta vs VFL): {meta_acc:.4f}")
print(f"[META] KL Divergence: {kl_div:.4f}")
print(f"[META] Logit MSE: {logit_mse:.4f}")

if meta_acc > 0.70:
    print(f" Good agreement! Meta-model successfully mimics VFL model.")
elif meta_acc > 0.50:
    print(f"Moderate agreement. Consider increasing meta-model capacity or training epochs.")
else:
    print(f"Low agreement. Check if VFL model is stable or increase meta-model capacity.")
# -----------------------------

# 9.5. Save Best Model and Metadata for Prediction
# -----------------------------
# joblib and Path are now imported at the top (cell 0)

# Create model directory if it doesn't exist
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

print("="*80)
print("SAVING BEST MODEL AND METADATA")
print("="*80)

# Save VFL model (best model is already loaded after training)
vfl_model_path = MODEL_DIR / "vfl_model_best.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_dims': [len(agent1_features), len(agent2_features), len(agent3_features)],
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes
    },
    'best_epoch': best_epoch,
    'best_val_f1': best_val_f1,
    'best_val_loss': best_val_loss
}, vfl_model_path)
print(f"✓ VFL model saved to {vfl_model_path}")

# Save meta-model
meta_model_path = MODEL_DIR / "meta_model_best.pth"
torch.save({
    'model_state_dict': meta_model.state_dict(),
    'model_config': {
        'in_dim': X_train_meta.shape[1],  # 192
        'num_classes': num_classes,
        'hidden_dim': 128
    }
}, meta_model_path)
print(f"✓ Meta-model saved to {meta_model_path}")

# Save scalers (using joblib for sklearn objects)
scaler1_path = MODEL_DIR / "scaler1.pkl"
scaler2_path = MODEL_DIR / "scaler2.pkl"
scaler3_path = MODEL_DIR / "scaler3.pkl"
joblib.dump(scaler1, scaler1_path)
joblib.dump(scaler2, scaler2_path)
joblib.dump(scaler3, scaler3_path)
print(f"✓ Scalers saved to {scaler1_path}, {scaler2_path}, {scaler3_path}")

# Save feature lists and label mapping
metadata = {
    'agent1_features': agent1_features,
    'agent2_features': agent2_features,
    'agent3_features': agent3_features,
    'label_mapping': label_mapping_dict,
    'label_mapping_reverse': {v: k for k, v in label_mapping_dict.items()},  # For reverse lookup
    'num_classes': num_classes,
    'embed_dim': embed_dim,
    'hidden_dim': hidden_dim,
    'agent_names': agent_names,
    'agent_domains': agent_domains,
    'agent_actions': agent_actions,
    'agent_action_mapping': agent_action_mapping,
    'model_info': {
        'vfl_model_path': str(vfl_model_path),
        'meta_model_path': str(meta_model_path),
        'scaler1_path': str(scaler1_path),
        'scaler2_path': str(scaler2_path),
        'scaler3_path': str(scaler3_path)
    },
    'training_info': {
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'best_val_loss': float(best_val_loss),
        'test_accuracy': float(acc) if 'acc' in globals() else None,
        'test_recall': float(rec) if 'rec' in globals() else None,
        'test_f1': float(f1) if 'f1' in globals() else None
    },
    'saved_timestamp': datetime.now().isoformat()
}

metadata_path = MODEL_DIR / "model_metadata.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✓ Metadata saved to {metadata_path}")

# Save SHAP explainer background (for faster prediction)
# This is the background data used for SHAP explanations
bg_size = min(100, X_train_meta.shape[0])
bg_idx = torch.randperm(X_train_meta.shape[0])[:bg_size]
shap_background = X_train_meta[bg_idx].detach().cpu().numpy()

shap_background_path = MODEL_DIR / "shap_background.npy"
np.save(shap_background_path, shap_background)
print(f"✓ SHAP background saved to {shap_background_path} (shape: {shap_background.shape})")

print("\n" + "="*80)
print("MODEL SAVE SUMMARY")
print("="*80)
print(f"All model files saved to: {MODEL_DIR.absolute()}")
print(f"\nFiles saved:")
print(f"  1. {vfl_model_path.name} - VFL model weights")
print(f"  2. {meta_model_path.name} - Meta-model weights")
print(f"  3. {scaler1_path.name}, {scaler2_path.name}, {scaler3_path.name} - Feature scalers")
print(f"  4. {metadata_path.name} - Model metadata and configuration")
print(f"  5. {shap_background_path.name} - SHAP explainer background")
print("\n" + "="*80)
print("Model ready for loading in prediction notebook!")
print("="*80)
# -----------------------------

# 10. SHAP on Party Meta-Features (Full Embeddings)
# -----------------------------
# Note: Meta-features are now 192-dim (64 per agent), so we compute SHAP on all 192 dims
# then aggregate per agent to get agent-level attributions

meta_model.eval()

bg_size = min(100, X_train_meta.shape[0])
bg_idx = torch.randperm(X_train_meta.shape[0])[:bg_size]
background = X_train_meta[bg_idx].detach().cpu().numpy()

def meta_predict(x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = meta_model(x_t)
        # Apply softmax to get probabilities for SHAP
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs

print(f"Computing SHAP on {X_train_meta.shape[1]}-dim meta-features (64 dims per agent)...")
explainer = shap.KernelExplainer(meta_predict, background)

n_explain = min(200, X_test_meta.shape[0])
X_explain = X_test_meta[:n_explain].detach().cpu().numpy()

# Compute SHAP values (will be 192-dim per sample)
shap_values = explainer.shap_values(X_explain, nsamples=200)

# Get predictions for selecting correct SHAP values
y_meta_pred_explain = torch.argmax(meta_model(torch.tensor(X_explain, dtype=torch.float32)), dim=1).cpu().numpy()

# Aggregate 192-dim SHAP values to 3 agent-level attributions
# Structure: first 64 dims = Agent 1, next 64 dims = Agent 2, last 64 dims = Agent 3
embed_dim = 64  # Each party has 64-dim embeddings
agent_dims = [embed_dim, embed_dim, embed_dim]  # [64, 64, 64]

def aggregate_shap_to_parties(shap_vals, n_samples):
    """
    Aggregate 192-dim SHAP values to 3 agent-level values by summing within each agent block.
    
    Args:
        shap_vals: SHAP values array, shape can be:
            - List of arrays: [num_classes, n_samples, 192] or [n_samples, 192, num_classes]
            - Single array: [n_samples, 192] or [num_classes, n_samples, 192]
        n_samples: Number of samples
    
    Returns:
        Aggregated SHAP values: [n_samples, 3] (one value per agent)
    """
    # Handle list format (multi-class)
    if isinstance(shap_vals, list):
        # For each class, aggregate to parties
        aggregated = []
        for class_idx in range(len(shap_vals)):
            class_shap = shap_vals[class_idx]  # [n_samples, 192]
            if len(class_shap.shape) == 2 and class_shap.shape[1] == embed_dim * 3:
                # Sum within each agent block
                agent_shap = np.zeros((n_samples, 3))
                for i in range(3):
                    start_idx = i * embed_dim
                    end_idx = (i + 1) * embed_dim
                    agent_shap[:, i] = class_shap[:, start_idx:end_idx].sum(axis=1)
                aggregated.append(agent_shap)
            else:
                # Fallback: if shape doesn't match, try to reshape
                aggregated.append(class_shap)
        return aggregated
    else:
        # Single array format
        shap_vals = np.asarray(shap_vals)
        if len(shap_vals.shape) == 2:
            # [n_samples, 192] - aggregate directly
            if shap_vals.shape[1] == embed_dim * 3:
                agent_shap = np.zeros((n_samples, 3))
                for i in range(3):
                    start_idx = i * embed_dim
                    end_idx = (i + 1) * embed_dim
                    agent_shap[:, i] = shap_vals[:, start_idx:end_idx].sum(axis=1)
                return agent_shap
        elif len(shap_vals.shape) == 3:
            # [num_classes, n_samples, 192] or [n_samples, 192, num_classes]
            if shap_vals.shape[0] == num_classes and shap_vals.shape[2] == embed_dim * 3:
                # [num_classes, n_samples, 192]
                aggregated = []
                for class_idx in range(num_classes):
                    class_shap = shap_vals[class_idx]  # [n_samples, 192]
                    agent_shap = np.zeros((n_samples, 3))
                    for i in range(3):
                        start_idx = i * embed_dim
                        end_idx = (i + 1) * embed_dim
                        agent_shap[:, i] = class_shap[:, start_idx:end_idx].sum(axis=1)
                    aggregated.append(agent_shap)
                return aggregated
            elif shap_vals.shape[0] == n_samples and shap_vals.shape[1] == embed_dim * 3:
                # [n_samples, 192, num_classes] - transpose first
                shap_vals = shap_vals.transpose(2, 0, 1)  # [num_classes, n_samples, 192]
                aggregated = []
                for class_idx in range(num_classes):
                    class_shap = shap_vals[class_idx]  # [n_samples, 192]
                    agent_shap = np.zeros((n_samples, 3))
                    for i in range(3):
                        start_idx = i * embed_dim
                        end_idx = (i + 1) * embed_dim
                        agent_shap[:, i] = class_shap[:, start_idx:end_idx].sum(axis=1)
                    aggregated.append(agent_shap)
                return aggregated
    
    # Fallback: return as-is if shape doesn't match expected
    print(f"Warning: Unexpected SHAP shape {shap_vals.shape}, returning as-is")
    return shap_vals

# Aggregate SHAP values to party level
shap_values_aggregated = aggregate_shap_to_parties(shap_values, n_explain)

# Handle SHAP output format - aggregated values should already be [n_explain, 3] or list of [n_explain, 3]
if isinstance(shap_values_aggregated, list):
    # Multi-class: SHAP returns list where each element is [n_explain, 3] for that class
    # Select SHAP values for predicted class for each sample
    shap_values_selected = []
    for i in range(n_explain):
        pred_class = int(y_meta_pred_explain[i])
        if pred_class < len(shap_values_aggregated):
            shap_values_selected.append(shap_values_aggregated[pred_class][i])
        else:
            # Fallback: use first class if prediction is out of bounds
            shap_values_selected.append(shap_values_aggregated[0][i])
    shap_values = np.array(shap_values_selected)  # [n_explain, 3]
else:
    # Already aggregated to [n_explain, 3]
    shap_values = np.asarray(shap_values_aggregated)
    if len(shap_values.shape) == 2 and shap_values.shape[1] == 3:
        # Already in correct format [n_explain, 3]
        pass
    elif len(shap_values.shape) == 3:
        # Could be [num_classes, n_explain, 3] - select by predicted class
        if shap_values.shape[0] == num_classes and shap_values.shape[2] == 3:
            shap_values_selected = []
            for i in range(n_explain):
                pred_class = int(y_meta_pred_explain[i])
                if pred_class < shap_values.shape[0]:
                    shap_values_selected.append(shap_values[pred_class, i, :])
                else:
                    shap_values_selected.append(shap_values[0, i, :])
            shap_values = np.array(shap_values_selected)  # [n_explain, 3]
        else:
            print(f"Warning: Unexpected aggregated SHAP shape {shap_values.shape}")
            # Try to extract
            shap_values = shap_values.reshape(-1, 3)[:n_explain]
    else:
        print(f"Warning: Unexpected aggregated SHAP shape {shap_values.shape}, attempting to reshape...")
        shap_values = shap_values.reshape(n_explain, -1)
        if shap_values.shape[1] > 3:
            # Sum to get 3 party values if needed
            agent_size = shap_values.shape[1] // 3
            shap_values = shap_values.reshape(n_explain, 3, agent_size).sum(axis=2)

print(f"SHAP values final shape: {shap_values.shape} (should be [{n_explain}, 3])")
print(f"Aggregated from 192-dim meta-features to 3 agent-level attributions")
# -----------------------------

# 11. Aggregate SHAP for Multi-Class
# -----------------------------
phi = shap_values  # [n_explain, 3]
phi_abs = np.abs(phi)

# Global mean |SHAP| and percentage per agent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mean_phi_abs = phi_abs.mean(axis=0)
mean_phi_pct = mean_phi_abs / mean_phi_abs.sum()

print("\n=== Global SHAP Agent Attributions ===")
global_rows = []
for i, name in enumerate(agent_names):
    m_abs = float(mean_phi_abs[i])
    m_pct = float(mean_phi_pct[i])
    print(f"{name}: mean |SHAP| = {m_abs:.6f}, mean contribution = {m_pct*100:5.2f}%")
    global_rows.append({
        "Agent": name,
        "Domain": agent_domains[i],
        "Feature_Group": agent_feature_groups[i],
        "Mean_abs_SHAP_All": m_abs,
        "Mean_contrib_All": m_pct,
    })
global_df = pd.DataFrame(global_rows)
global_filename = f"outputs/vfl_shap_global_summary_{timestamp}.csv"
global_df.to_csv(global_filename, index=False)

# Per-class SHAP analysis
y_test_np = y_test.cpu().numpy()
y_explain = y_test_np[:n_explain]

print("\n=== SHAP Agent Attributions by Class ===")
for class_idx in range(num_classes):
    class_name = label_mapping_dict.get(class_idx, f"Class_{class_idx}")
    class_mask = (y_explain == class_idx)
    
    if class_mask.sum() == 0:
        continue
    
    phi_class = phi_abs[class_mask]
    mean_phi_class = phi_class.mean(axis=0)
    mean_pct_class = mean_phi_class / mean_phi_class.sum()
    
    print(f"\n{class_name} (class {class_idx}):")
    class_rows = []
    for i, name in enumerate(agent_names):
        c_abs = float(mean_phi_class[i])
        c_pct = float(mean_pct_class[i])
        print(f"  {name}: mean |SHAP| = {c_abs:.6f}, mean contribution = {c_pct*100:5.2f}%")
        class_rows.append({
            "Agent": name,
            "Domain": agent_domains[i],
            "Feature_Group": agent_feature_groups[i],
            "Class": class_name,
            "Class_Index": class_idx,
            "Mean_abs_SHAP": c_abs,
            "Mean_contrib": c_pct,
        })
    
    class_df = pd.DataFrame(class_rows)
    class_filename = f"summary/vfl_shap_{class_name.lower()}_summary_{timestamp}.csv"
    class_df.to_csv(class_filename, index=False)

# Find dominant agent per class
print("\n=== Dominant Agent per Class ===")
for class_idx in range(num_classes):
    class_name = label_mapping_dict.get(class_idx, f"Class_{class_idx}")
    class_mask = (y_explain == class_idx)
    
    if class_mask.sum() == 0:
        continue
    
    phi_class = phi_abs[class_mask]
    mean_phi_class = phi_class.mean(axis=0)
    mean_pct_class = mean_phi_class / mean_phi_class.sum()
    
    top_agent_idx = int(np.argmax(mean_pct_class))
    top_agent_name = agent_names[top_agent_idx]
    top_agent_share = float(mean_pct_class[top_agent_idx]) * 100.0
    
    print(f"{class_name}: {top_agent_name} ({top_agent_share:.2f}%)")
# -----------------------------