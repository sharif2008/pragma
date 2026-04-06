"""Standalone VFL SHAP prediction (merged from VFL_SHAP_Prediction.ipynb)."""
# VFL SHAP - Prediction Notebook
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Import utility functions
from app.notebook_runtime.vfl_utils import simplify_label, FIXED_PARTY_NAMES, FIXED_AGENT_NAMES, get_agent_names

# Import model classes
from app.notebook_runtime.model_utils import VFLModel, AgentMetaModel

# -----------------------------


# 1. Load Saved Model and Metadata
# -----------------------------
# joblib and Path are imported in cell 0

# Model directory
MODEL_DIR = Path("model")

print("="*80)
print("LOADING SAVED MODEL AND METADATA")
print("="*80)

# Check if model directory exists
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found! Please train the model first using VFL_SHAP_MultiClass.ipynb")

# Load metadata first (contains all configuration)
metadata_path = MODEL_DIR / "model_metadata.json"
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"✓ Metadata loaded from {metadata_path}")

# Extract configuration from metadata
agent1_features = metadata.get('agent1_features') or metadata.get('party1_features')
agent2_features = metadata.get('agent2_features') or metadata.get('party2_features')
agent3_features = metadata.get('agent3_features') or metadata.get('party3_features')
label_mapping_dict = {int(k): v for k, v in metadata['label_mapping'].items()}  # Convert keys back to int
num_classes = metadata['num_classes']
embed_dim = metadata['embed_dim']
hidden_dim = metadata['hidden_dim']
# Get agent names from metadata, fallback to fixed agent names
agent_names = metadata.get('agent_names') or metadata.get('party_names')
if not agent_names:
    # Use fixed agent names from vfl_utils as fallback
    agent_names = get_agent_names()
agent_domains = metadata.get('agent_domains', metadata.get('party_domains'))
agent_actions = metadata.get('agent_actions', metadata.get('party_actions'))
agent_action_mapping = metadata.get('agent_action_mapping', metadata.get('party_action_mapping'))

print(f"✓ Configuration loaded: {num_classes} classes, embed_dim={embed_dim}, hidden_dim={hidden_dim}")

# Load scalers
scaler1_path = MODEL_DIR / "scaler1.pkl"
scaler2_path = MODEL_DIR / "scaler2.pkl"
scaler3_path = MODEL_DIR / "scaler3.pkl"

scaler1 = joblib.load(scaler1_path)
scaler2 = joblib.load(scaler2_path)
scaler3 = joblib.load(scaler3_path)
print(f"✓ Scalers loaded from {scaler1_path}, {scaler2_path}, {scaler3_path}")

# Model classes are now imported from model_utils.py
# LocalEncoder, ActiveClassifier, VFLModel, and AgentMetaModel are available

# Load VFL model
vfl_model_path = MODEL_DIR / "vfl_model_best.pth"
vfl_checkpoint = torch.load(vfl_model_path, map_location='cpu')
vfl_config = vfl_checkpoint['model_config']

model = VFLModel(
    input_dims=vfl_config['input_dims'],
    embed_dim=vfl_config['embed_dim'],
    num_classes=vfl_config['num_classes'],
    hidden_dim=vfl_config['hidden_dim']
)
model.load_state_dict(vfl_checkpoint['model_state_dict'])
model.eval()
print(f"✓ VFL model loaded from {vfl_model_path}")
print(f"  Best epoch: {vfl_checkpoint['best_epoch']}, Val F1: {vfl_checkpoint['best_val_f1']:.4f}")

# Load meta-model
meta_model_path = MODEL_DIR / "meta_model_best.pth"
meta_checkpoint = torch.load(meta_model_path, map_location='cpu')
meta_config = meta_checkpoint['model_config']

meta_model = AgentMetaModel(
    in_dim=meta_config['in_dim'],
    num_classes=meta_config['num_classes'],
    hidden_dim=meta_config['hidden_dim']
)
meta_model.load_state_dict(meta_checkpoint['model_state_dict'])
meta_model.eval()
print(f"✓ Meta-model loaded from {meta_model_path}")

# Load SHAP background
shap_background_path = MODEL_DIR / "shap_background.npy"
shap_background = np.load(shap_background_path)
print(f"✓ SHAP background loaded from {shap_background_path} (shape: {shap_background.shape})")

# Initialize SHAP explainer
def meta_predict(x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = meta_model(x_t)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs

explainer = shap.KernelExplainer(meta_predict, shap_background)
print(f"✓ SHAP explainer initialized")

print("\n" + "="*80)
print("MODEL LOADING SUMMARY")
print("="*80)
print(f"✓ All components loaded successfully!")
if agent1_features and agent2_features and agent3_features:
    print(f"  - VFL Model: {len(agent1_features) + len(agent2_features) + len(agent3_features)} total features")
else:
    print(f"  - VFL Model: Feature lists not available in metadata")
print(f"  - Meta Model: {meta_config['in_dim']} input dims, {num_classes} classes")
print(f"  - Label mapping: {len(label_mapping_dict)} classes")
if agent_names:
    print(f"  - Parties: {len(agent_names)}")
    for i, name in enumerate(agent_names):
        feat_list = [agent1_features, agent2_features, agent3_features][i] if i < 3 else []
        feat_count = len(feat_list) if feat_list else 0
        print(f"    Agent {i+1}: {name} ({feat_count} features)")
else:
    print(f"  - Parties: Agent names not available in metadata")
print("="*80)
print("Model ready for prediction!")
print("="*80)
# -----------------------------

# 2. Load Sample Data and Predict
# -----------------------------
# Ensure agent_names is initialized (in case cell 1 wasn't run or agent_names is None)
try:
    if not agent_names:
        agent_names = get_agent_names()
except NameError:
    # agent_names doesn't exist, initialize it
    agent_names = get_agent_names()

# Configuration
INPUTS_DIR = Path("inputs")
INPUTS_DIR.mkdir(exist_ok=True)
SAMPLE_CSV_PATH = INPUTS_DIR / "sample.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("LOADING SAMPLE DATA AND PREDICTING")
print("="*80)

# Load sample CSV file
if not SAMPLE_CSV_PATH.exists():
    raise FileNotFoundError(f"Sample CSV file not found: {SAMPLE_CSV_PATH}\nPlease ensure sample.csv exists in the 'inputs' folder.")

sample_df = pd.read_csv(SAMPLE_CSV_PATH)
print(f"✓ Loaded {len(sample_df)} rows from {SAMPLE_CSV_PATH}")

# Drop unnecessary columns
sample_df = sample_df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"], errors="ignore")

# Validate that all required features exist in sample data
print("\nValidating required features...")
all_required_features = agent1_features + agent2_features + agent3_features
missing_features = [feat for feat in all_required_features if feat not in sample_df.columns]

if missing_features:
    print(f"❌ ERROR: {len(missing_features)} required features are missing from sample.csv:")
    for feat in missing_features[:20]:  # Show first 20 missing features
        print(f"  - {feat}")
    if len(missing_features) > 20:
        print(f"  ... and {len(missing_features) - 20} more")
    print(f"\nRequired features: {len(all_required_features)}")
    print(f"Found in sample.csv: {len(all_required_features) - len(missing_features)}")
    print(f"Available columns in sample.csv: {len(sample_df.columns)}")
    print(f"\nSample of available columns (first 10):")
    for col in sample_df.columns[:10]:
        print(f"  - {col}")
    if len(sample_df.columns) > 10:
        print(f"  ... and {len(sample_df.columns) - 10} more")
    print(f"\nPlease ensure sample.csv contains all features used during training.")
    print(f"Common issues:")
    print(f"  1. Column name mismatches (case sensitivity, spaces, etc.)")
    print(f"  2. Missing feature columns")
    print(f"  3. Different dataset format")
    raise KeyError(f"Missing {len(missing_features)} required features in sample.csv. See list above.")

print(f"✓ All {len(all_required_features)} required features found in sample.csv")

# Helper function to partition and preprocess
def partition_and_preprocess_sample(sample_df, agent1_features, agent2_features, agent3_features, 
                                     scaler1, scaler2, scaler3):
    """Partition and preprocess sample data same as training"""
    # Validate features exist (should already be checked, but double-check for safety)
    for agent_idx, agent_feats in enumerate([agent1_features, agent2_features, agent3_features], 1):
        missing = [f for f in agent_feats if f not in sample_df.columns]
        if missing:
            raise KeyError(f"Agent {agent_idx} missing features: {missing[:5]}")
    
    X1_raw = sample_df[agent1_features].values
    X2_raw = sample_df[agent2_features].values
    X3_raw = sample_df[agent3_features].values
    
    X1 = torch.tensor(scaler1.transform(X1_raw), dtype=torch.float32)
    X2 = torch.tensor(scaler2.transform(X2_raw), dtype=torch.float32)
    X3 = torch.tensor(scaler3.transform(X3_raw), dtype=torch.float32)
    
    return X1, X2, X3

# Preprocess sample data
X1_sample, X2_sample, X3_sample = partition_and_preprocess_sample(
    sample_df, agent1_features, agent2_features, agent3_features, scaler1, scaler2, scaler3
)
x_sample_parts = [X1_sample, X2_sample, X3_sample]
print(f"✓ Preprocessed {len(x_sample_parts[0])} samples")

# Check for labels (optional)
y_sample = None
if "label" in sample_df.columns:
    try:
        sample_df['label_simplified'] = sample_df['label'].apply(simplify_label)
        sample_df['label_numeric'] = sample_df['label_simplified'].map(
            {v: k for k, v in label_mapping_dict.items()}
        )
        y_sample = torch.tensor(sample_df['label_numeric'].values, dtype=torch.long)
        print(f"✓ Labels found: {len(y_sample)} samples with ground truth")
    except Exception as e:
        print(f"⚠ Could not process labels: {e}")

# 3. Predict on All Samples and Save Results
# -----------------------------
print("\n" + "="*80)
print("PREDICTING ON ALL SAMPLES")
print("="*80)

n_samples = len(x_sample_parts[0])
results = []

print(f"Processing {n_samples} samples...")
model.eval()
meta_model.eval()

for idx in range(n_samples):
    if (idx + 1) % 50 == 0 or idx == 0:
        print(f"  Processing sample {idx + 1}/{n_samples}...")
    
    # Get sample
    X1_s = x_sample_parts[0][idx:idx+1]
    X2_s = x_sample_parts[1][idx:idx+1]
    X3_s = x_sample_parts[2][idx:idx+1]
    
    # Get true label if available
    if y_sample is not None:
        true_label_idx = int(y_sample[idx].item())
        true_label = label_mapping_dict.get(true_label_idx, f"Class_{true_label_idx}")
    else:
        true_label_idx = None
        true_label = "UNKNOWN"
    
    # Predict
    with torch.no_grad():
        embeddings = model.get_agent_embeddings([X1_s, X2_s, X3_s])
        X_meta = torch.cat(embeddings, dim=1)
        logits = meta_model(X_meta)
        probs = torch.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class_idx].item()
    
    predicted_label = label_mapping_dict.get(predicted_class_idx, f"Class_{predicted_class_idx}")
    
    # Get all probabilities
    all_probs = {label_mapping_dict.get(i, f"Class_{i}"): float(probs[0, i].item()) 
                 for i in range(num_classes)}
    
    # Compute SHAP (agent-level)
    X_meta_np = X_meta.detach().cpu().numpy()
    shap_vals_sample = explainer.shap_values(X_meta_np, nsamples=50)
    
    if isinstance(shap_vals_sample, list):
        shap_vals = shap_vals_sample[predicted_class_idx][0]
    else:
        shap_vals = shap_vals_sample[0]
    
    # Aggregate to party level
    embed_dim = 64
    agent_shap = []
    for i in range(3):
        start = i * embed_dim
        end = (i + 1) * embed_dim
        agent_shap.append(float(np.sum(np.abs(shap_vals[start:end]))))
    
    total_shap = sum(agent_shap)
    agent_shap_pct = [p / total_shap if total_shap > 0 else 0 for p in agent_shap]
    dominant_agent_idx = int(np.argmax(agent_shap))
    dominant_agent = agent_names[dominant_agent_idx]
    
    # Compute feature-level contributions per agent using gradient × input
    # (each feature gets a different contribution; no KernelExplainer / equal fallback)
    feature_shap_contributions = {}
    party_feature_lists = [agent1_features, agent2_features, agent3_features]
    party_data = [X1_s, X2_s, X3_s]
    
    model.eval()
    meta_model.eval()
    for agent_idx in range(3):
        party_name = agent_names[agent_idx]
        party_feat_list = party_feature_lists[agent_idx]
        X_agent = party_data[agent_idx]
        # One agent input with grad for attribution
        X_agent_var = X_agent.clone().detach().requires_grad_(True)
        parts = [X1_s, X2_s, X3_s]
        parts[agent_idx] = X_agent_var
        # Compute embeddings without no_grad so gradients flow (get_agent_embeddings uses torch.no_grad())
        embeddings = [model.encoders[i](parts[i]) for i in range(3)]
        X_meta = torch.cat(embeddings, dim=1)
        logits = meta_model(X_meta)
        target = logits[0, predicted_class_idx]
        target.backward()
        grad = X_agent_var.grad
        if grad is None:
            grad = torch.zeros_like(X_agent_var)
        contrib = (grad * X_agent_var).squeeze(0).detach().cpu().numpy()
        total_abs = float(np.sum(np.abs(contrib)))
        feature_contribs = {}
        for feat_idx, feat_name in enumerate(party_feat_list):
            if feat_idx < len(contrib):
                shap_val = float(contrib[feat_idx])
                abs_shap_val = float(np.abs(shap_val))
                pct_contrib = (abs_shap_val / total_abs * 100.0) if total_abs > 1e-12 else 0.0
                feature_contribs[feat_name] = {
                    "shap_value": shap_val,
                    "abs_shap_value": abs_shap_val,
                    "pct_contribution": pct_contrib
                }
        feature_shap_contributions[party_name] = feature_contribs
    
    # Store result with full structure
    results.append({
        "sample_id": idx,
        "true_label": true_label,
        "true_label_idx": int(true_label_idx) if true_label_idx is not None else None,
        "predicted_label": predicted_label,
        "predicted_label_idx": int(predicted_class_idx),
        "confidence": float(confidence),
        "all_probabilities": all_probs,
        "is_correct": (true_label == predicted_label) if true_label != "UNKNOWN" else None,
        "shap_explanation": {
            "party_contributions": {
                agent_names[0]: float(agent_shap[0]),
                agent_names[1]: float(agent_shap[1]),
                agent_names[2]: float(agent_shap[2])
            },
            "party_contributions_pct": {
                agent_names[0]: float(agent_shap_pct[0]),
                agent_names[1]: float(agent_shap_pct[1]),
                agent_names[2]: float(agent_shap_pct[2])
            },
            "dominant_agent": dominant_agent,
            "dominant_agent_idx": int(dominant_agent_idx),
            "dominant_contribution": float(agent_shap[dominant_agent_idx]),
            "dominant_contribution_pct": float(agent_shap_pct[dominant_agent_idx]),
            "total_contribution": float(total_shap),
            "feature_contributions": feature_shap_contributions
        },
        "timestamp": datetime.now().isoformat()
    })

print(f"✓ Completed predictions for {len(results)} samples")

# Calculate accuracy if labels available
if y_sample is not None:
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    print(f"✓ Accuracy: {accuracy:.2%}")

# 4. Save Results to Output Folder
# -----------------------------
print("\n" + "="*80)
print("SAVING RESULTS TO OUTPUT FOLDER")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save predictions summary CSV (flattened for easier reading)
summary_data = []
for r in results:
    summary_data.append({
        "sample_id": r["sample_id"],
        "true_label": r["true_label"],
        "predicted_label": r["predicted_label"],
        "confidence": r["confidence"],
        "is_correct": r["is_correct"],
        "dominant_agent": r["shap_explanation"]["dominant_agent"],
        "dominant_contribution_pct": r["shap_explanation"]["dominant_contribution_pct"],
        "party1_contrib_pct": r["shap_explanation"]["party_contributions_pct"][agent_names[0]],
        "party2_contrib_pct": r["shap_explanation"]["party_contributions_pct"][agent_names[1]],
        "party3_contrib_pct": r["shap_explanation"]["party_contributions_pct"][agent_names[2]]
    })
results_df = pd.DataFrame(summary_data)
summary_file = OUTPUT_DIR / f"predictions_{timestamp}.csv"
results_df.to_csv(summary_file, index=False)
print(f"✓ Predictions saved to: {summary_file}")

# Save detailed JSON
detailed_file = OUTPUT_DIR / f"predictions_detailed_{timestamp}.json"
with open(detailed_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✓ Detailed results saved to: {detailed_file}")

# Save decision summary
decision_summary = {
    "timestamp": datetime.now().isoformat(),
    "total_samples": len(results),
    "predictions": {
        label: sum(1 for r in results if r["predicted_label"] == label)
        for label in label_mapping_dict.values()
    },
    "dominant_agents": {
        agent: sum(1 for r in results if (r.get("shap_explanation", {}).get("dominant_agent") or r.get("shap_explanation", {}).get("dominant_party")) == agent)
        for agent in agent_names
    }
}

if y_sample is not None:
    decision_summary["accuracy"] = accuracy
    decision_summary["correct_predictions"] = sum(1 for r in results if r["is_correct"])

decision_file = OUTPUT_DIR / f"decision_summary_{timestamp}.json"
with open(decision_file, "w", encoding="utf-8") as f:
    json.dump(decision_summary, f, indent=2, ensure_ascii=False)
print(f"✓ Decision summary saved to: {decision_file}")

print("\n" + "="*80)
print("PREDICTION COMPLETE")
print("="*80)
print(f"All results saved to: {OUTPUT_DIR}")
print(f"  - Summary CSV: {summary_file.name}")
print(f"  - Detailed JSON: {detailed_file.name}")
print(f"  - Decision Summary: {decision_file.name}")
print("="*80)
# -----------------------------

# -----------------------------