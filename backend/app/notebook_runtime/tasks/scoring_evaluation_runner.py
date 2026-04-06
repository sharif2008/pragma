"""Standalone scoring evaluation (merged from Scoring_Evaluation.ipynb)."""
# Import required libraries
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scoring libraries
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
import nltk

import json
from collections import defaultdict

from app.notebook_runtime.storage_paths import AGENTIC_FEATURES_JSON, ATTACK_OPTIONS_JSON

# --- Load attack / agent catalogs from backend/storage ---
with open(ATTACK_OPTIONS_JSON, "r", encoding="utf-8") as f:
    attack_options_kb = json.load(f)["attacks"]

with open(AGENTIC_FEATURES_JSON, "r", encoding="utf-8") as f:
    agents_kb = json.load(f)["agents"]

# Normalize action strings for matching
def _norm_action(s: str) -> str:
    return " ".join(s.lower().strip().split())

# Build: action -> agents that can execute it (based on action_capabilities)
action_to_agents = defaultdict(list)
for agent_name, agent_info in agents_kb.items():
    for a in agent_info.get("action_capabilities", []):
        action_to_agents[_norm_action(a)].append(agent_name)

# Optional: synonyms so "limit rate" matches "rate limiting", etc.
# Keep synonyms SHORT and close to what your plan generator actually emits.
ACTION_SYNONYMS = {
    "rate limiting": ["limit rate", "rate limit"],
    "traffic scrubbing": ["enable scrubbing", "scrub traffic"],
    "blackhole routing": ["blackhole route", "blackhole"],
    "ip blocking": ["block ip", "ip block"],
    "acl update": ["update acl", "acl change"],
    "syn cookies": ["enable syn cookies", "syn cookie"],
    "connection limiting": ["limit connections", "conn limit"],
    "account lockout": ["lock account", "lockout"],
    "mfa enforce": ["enforce mfa", "require mfa"],
    "credential throttle": ["throttle login", "login throttle"],
    "fail2ban block": ["fail2ban", "ban ip"],
    "waf rules": ["waf rule", "apply waf"],
    "virtual patching": ["virtual patch", "hot patch"],
    "isolate service": ["isolate", "quarantine service"],
    "captcha challenge": ["captcha", "challenge"],
    "reputation filter": ["reputation", "filter reputation"],
    "js challenge": ["js challenge", "javascript challenge"],
    "tarpitting": ["tarpit", "slow scan"],
    "port hardening": ["harden ports", "close ports"],
    "scan threshold": ["threshold scan", "scan threshold"],
    "auto scale": ["autoscale", "scale out"],
    "log only": ["log", "logging"],
    "monitor": ["monitoring", "observe"],
}

def build_reference_text_for_label(label: str) -> str:
    """
    Create a deterministic 'gold' reference string for scoring,
    based on attack_options + which agents can execute which actions.
    """
    label = label.strip().upper()
    actions = attack_options_kb.get(label) or attack_options_kb.get("OTHERS", [])
    actions_norm = [_norm_action(a) for a in actions]

    # Assign each action to agents that can execute it (can be multiple).
    agent_to_actions = defaultdict(list)
    unassigned = []
    for a_raw, a in zip(actions, actions_norm):
        capable_agents = action_to_agents.get(a, [])
        if not capable_agents:
            # If you want STRICT no-unassigned, you can drop it or map to "Core" by policy.
            unassigned.append(a_raw)
            continue

        # Add canonical + synonyms into the agent bucket to maximize token overlap.
        expanded = [a_raw]
        expanded += ACTION_SYNONYMS.get(a, [])
        for ag in capable_agents:
            agent_to_actions[ag].extend(expanded)

    # Stable ordering for reproducible scores
    def uniq(seq):
        seen = set()
        out = []
        for x in seq:
            k = _norm_action(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    ran = ", ".join(uniq(agent_to_actions.get("RAN", []))) or "none"
    edge = ", ".join(uniq(agent_to_actions.get("Edge", []))) or "none"
    core = ", ".join(uniq(agent_to_actions.get("Core", []))) or "none"

    # One compact canonical string your metrics will compare against
    ref = (
        f"{label} response: "
        f"RAN actions: {ran}. "
        f"Edge actions: {edge}. "
        f"Core actions: {core}."
    )

    if unassigned:
        # Keep it visible during dev; if you dislike extra tokens, remove this line.
        ref += f" Unassigned actions: {', '.join(unassigned)}."

    return ref

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize SBERT model (load once, reuse)
print("Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Libraries imported successfully!")

# Configuration
RAG_DOCS_DIR = Path("RAG_docs/action_plans")
REFERENCE_TEXT = "Rate limiting on RAN. Rate limiting on Edge. Rate limiting on Core. Traffic scrubbing on Core. Blackhole routing on Core. ACL update on RAN. ACL update on Edge. ACL update on Core. IP blocking on RAN. IP blocking on Edge. IP blocking on Core."

print(f"Action plans directory: {RAG_DOCS_DIR}")

print(f"Reference text: {REFERENCE_TEXT}")

def extract_overall_reasoning(action_plan: Dict[str, Any]) -> str:
    """
    Extract overall_reasoning from an action plan.
    Returns the overall_reasoning text if available, otherwise empty string.
    """
    overall_reasoning = action_plan.get("overall_reasoning", "").strip()
    return overall_reasoning if overall_reasoning else ""


def load_comparison_files(directory: Path) -> List[Dict[str, Any]]:
    """Load all comparison JSON files from the directory."""
    comparison_files = list(directory.glob("comparison*.json"))
    print(f"Found {len(comparison_files)} comparison file(s)")
    
    loaded_data = []
    for file_path in comparison_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['file_path'] = str(file_path)
                data['file_name'] = file_path.name
                loaded_data.append(data)
                print(f"  - Loaded: {file_path.name}")
        except Exception as e:
            print(f"  - Error loading {file_path.name}: {e}")
    
    return loaded_data


def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate texts."""
    # Tokenize
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    # Use smoothing function to handle cases where n-grams don't match
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    return score


def calculate_rouge1_score(reference: str, candidate: str) -> float:
    """Calculate ROUGE-1 score only."""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure


def calculate_sbert_cosine(reference: str, candidate: str) -> float:
    """Calculate SBERT cosine similarity between reference and candidate texts."""
    # Get embeddings
    ref_embedding = sbert_model.encode([reference])
    cand_embedding = sbert_model.encode([candidate])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(ref_embedding, cand_embedding)[0][0]
    return float(cosine_sim)


def calculate_bertscore_scores(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """Calculate BERTScore (precision, recall, F1) for multiple reference-candidate pairs."""
    # BERTScore expects lists of strings
    P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
    
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }

print("Helper functions defined!")

# Load all comparison files
comparison_data = load_comparison_files(RAG_DOCS_DIR)

if not comparison_data:
    print("No comparison files found!")
else:
    print(f"\nTotal comparison files loaded: {len(comparison_data)}")

# Extract overall_reasoning from action plans and prepare for scoring
scoring_results = []

for comp_data in comparison_data:
    file_name = comp_data['file_name']
    sample_id = comp_data.get('sample_id', 'unknown')
    predicted_label = comp_data.get("prediction", {}).get("predicted_label", "OTHERS")
    reference_text = build_reference_text_for_label(predicted_label)
    REFERENCE_TEXT = reference_text

    
    # Extract with RAG overall_reasoning
    with_rag_plan = comp_data.get('action_plans', {}).get('with_rag', {})
    with_rag_text = extract_overall_reasoning(with_rag_plan)
    
    # Extract without RAG overall_reasoning
    without_rag_plan = comp_data.get('action_plans', {}).get('without_rag', {})
    without_rag_text = extract_overall_reasoning(without_rag_plan)
    
    print(f"\n{'='*60}")
    print(f"File: {file_name} (Sample ID: {sample_id})")
    print(f"{'='*60}")
    print(f"With RAG overall_reasoning: {with_rag_text}")
    print(f"Without RAG overall_reasoning: {without_rag_text}")
    
    # Calculate scores for with RAG
    if with_rag_text:
        bleu_with = calculate_bleu_score(REFERENCE_TEXT, with_rag_text)
        rouge1_with = calculate_rouge1_score(REFERENCE_TEXT, with_rag_text)
        sbert_cosine_with = calculate_sbert_cosine(REFERENCE_TEXT, with_rag_text)
        bertscore_with = calculate_bertscore_scores([REFERENCE_TEXT], [with_rag_text])
        
        scoring_results.append({
            'file_name': file_name,
            'sample_id': sample_id,
            'method': 'with_rag',
            'text': with_rag_text,
            'bleu': bleu_with,
            'rouge1': rouge1_with,
            'sbert_cosine': sbert_cosine_with,
            'bertscore_precision': bertscore_with['bertscore_precision'],
            'bertscore_recall': bertscore_with['bertscore_recall'],
            'bertscore_f1': bertscore_with['bertscore_f1']
        })
    
    # Calculate scores for without RAG
    if without_rag_text:
        bleu_without = calculate_bleu_score(REFERENCE_TEXT, without_rag_text)
        rouge1_without = calculate_rouge1_score(REFERENCE_TEXT, without_rag_text)
        sbert_cosine_without = calculate_sbert_cosine(REFERENCE_TEXT, without_rag_text)
        bertscore_without = calculate_bertscore_scores([REFERENCE_TEXT], [without_rag_text])
        
        scoring_results.append({
            'file_name': file_name,
            'sample_id': sample_id,
            'method': 'without_rag',
            'text': without_rag_text,
            'bleu': bleu_without,
            'rouge1': rouge1_without,
            'sbert_cosine': sbert_cosine_without,
            'bertscore_precision': bertscore_without['bertscore_precision'],
            'bertscore_recall': bertscore_without['bertscore_recall'],
            'bertscore_f1': bertscore_without['bertscore_f1']
        })

print(f"\n\nTotal scoring results: {len(scoring_results)}")

# Create DataFrame for easier analysis
df_results = pd.DataFrame(scoring_results)

if not df_results.empty:
    print("Scoring Results DataFrame:")
    print(df_results[['file_name', 'method', 'bleu', 'rouge1', 'sbert_cosine', 
                      'bertscore_precision', 'bertscore_recall', 'bertscore_f1']].to_string())
else:
    print("No results to display!")

# Calculate average scores by method (with_rag vs without_rag)
if not df_results.empty:
    avg_scores = df_results.groupby('method').agg({
        'bleu': 'mean',
        'rouge1': 'mean',
        'sbert_cosine': 'mean',
        'bertscore_precision': 'mean',
        'bertscore_recall': 'mean',
        'bertscore_f1': 'mean'
    }).round(4)
    
    print("\nAverage Scores by Method:")
    print("="*60)
    print(avg_scores.to_string())
    
    # Also calculate overall averages
    overall_avg = df_results[['bleu', 'rouge1', 'sbert_cosine', 
                               'bertscore_precision', 'bertscore_recall', 'bertscore_f1']].mean()
    
    print("\n\nOverall Average Scores (across all methods):")
    print("="*60)
    for metric, value in overall_avg.items():
        print(f"{metric:25s}: {value:.4f}")
else:
    print("No results to calculate averages!")

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# Create visualizations
if not df_results.empty and len(df_results) > 0:
    # Prepare data for plotting
    metrics = ['bleu', 'rouge1', 'sbert_cosine', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    metric_labels = ['BLEU', 'ROUGE-1', 'SBERT Cosine', 'BERTScore-P', 'BERTScore-R', 'BERTScore-F1']
    
    # Calculate averages by method
    avg_scores = df_results.groupby('method')[metrics].mean()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Action Plan Scoring Comparison: With RAG vs Without RAG', fontsize=16, fontweight='bold')
    
    # 1. All metrics comparison
    ax1 = axes[0, 0]
    x = np.arange(len(metrics))
    width = 0.35
    
    if 'with_rag' in avg_scores.index:
        with_rag_values = [avg_scores.loc['with_rag', metric] for metric in metrics]
        ax1.bar(x - width/2, with_rag_values, width, label='With RAG', alpha=0.8)
    
    if 'without_rag' in avg_scores.index:
        without_rag_values = [avg_scores.loc['without_rag', metric] for metric in metrics]
        ax1.bar(x + width/2, without_rag_values, width, label='Without RAG', alpha=0.8)
    
    ax1.set_xlabel('Metrics', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. BLEU, ROUGE-1, and SBERT Cosine
    ax2 = axes[0, 1]
    text_metrics = ['bleu', 'rouge1', 'sbert_cosine']
    text_labels = ['BLEU', 'ROUGE-1', 'SBERT Cosine']
    x2 = np.arange(len(text_metrics))
    
    if 'with_rag' in avg_scores.index:
        with_rag_values = [avg_scores.loc['with_rag', metric] for metric in text_metrics]
        ax2.bar(x2 - width/2, with_rag_values, width, label='With RAG', alpha=0.8)
    
    if 'without_rag' in avg_scores.index:
        without_rag_values = [avg_scores.loc['without_rag', metric] for metric in text_metrics]
        ax2.bar(x2 + width/2, without_rag_values, width, label='Without RAG', alpha=0.8)
    
    ax2.set_xlabel('Metrics', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Text Similarity Metrics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(text_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. BERTScore components
    ax3 = axes[1, 0]
    bert_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    bert_labels = ['Precision', 'Recall', 'F1']
    x3 = np.arange(len(bert_metrics))
    
    if 'with_rag' in avg_scores.index:
        with_rag_values = [avg_scores.loc['with_rag', metric] for metric in bert_metrics]
        ax3.bar(x3 - width/2, with_rag_values, width, label='With RAG', alpha=0.8)
    
    if 'without_rag' in avg_scores.index:
        without_rag_values = [avg_scores.loc['without_rag', metric] for metric in bert_metrics]
        ax3.bar(x3 + width/2, without_rag_values, width, label='Without RAG', alpha=0.8)
    
    ax3.set_xlabel('Metrics', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('BERTScore Components', fontsize=12, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(bert_labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Overall comparison (stacked or grouped)
    ax4 = axes[1, 1]
    methods = ['With RAG', 'Without RAG'] if 'with_rag' in avg_scores.index and 'without_rag' in avg_scores.index else []
    if methods:
        overall_metrics = ['bleu', 'rouge1', 'sbert_cosine', 'bertscore_f1']
        overall_labels = ['BLEU', 'ROUGE-1', 'SBERT Cosine', 'BERTScore-F1']
        
        x4 = np.arange(len(methods))
        bottom = np.zeros(len(methods))
        
        for i, metric in enumerate(overall_metrics):
            values = []
            if 'with_rag' in avg_scores.index:
                values.append(avg_scores.loc['with_rag', metric])
            if 'without_rag' in avg_scores.index:
                values.append(avg_scores.loc['without_rag', metric])
            
            if values:
                ax4.bar(methods, values, label=overall_labels[i], alpha=0.8, bottom=bottom[:len(values)])
                bottom[:len(values)] += np.array(values)
        
        ax4.set_ylabel('Cumulative Score', fontsize=11)
        ax4.set_title('Overall Score Comparison', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualizations created successfully!")
else:
    print("No data available for visualization!")

# Display individual file scores
if not df_results.empty:
    print("\nIndividual File Scores:")
    print("="*80)
    
    for file_name in df_results['file_name'].unique():
        file_data = df_results[df_results['file_name'] == file_name]
        print(f"\nFile: {file_name}")
        print("-" * 80)
        
        for _, row in file_data.iterrows():
            print(f"\nMethod: {row['method'].upper()}")
            print(f"  BLEU:              {row['bleu']:.4f}")
            print(f"  ROUGE-1:           {row['rouge1']:.4f}")
            print(f"  SBERT Cosine:      {row['sbert_cosine']:.4f}")
            print(f"  BERTScore-P:       {row['bertscore_precision']:.4f}")
            print(f"  BERTScore-R:       {row['bertscore_recall']:.4f}")
            print(f"  BERTScore-F1:      {row['bertscore_f1']:.4f}")
            print(f"  Overall Reasoning: {row['text'][:100]}..." if len(row['text']) > 100 else f"  Overall Reasoning: {row['text']}")
else:
    print("No results to display!")

# Save detailed results to JSON
if scoring_results:
    # Create scoring directory under RAG_docs
    scoring_dir = Path("RAG_docs/scoring")
    scoring_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = scoring_dir / "scoring_results.json"
    
    # Prepare output data
    output_data = {
        'reference_text': REFERENCE_TEXT,
        'total_files': len(comparison_data),
        'timestamp': pd.Timestamp.now().isoformat(),
        'individual_scores': scoring_results,
        'average_scores': {}
    }
    
    # Add average scores
    if not df_results.empty:
        for method in df_results['method'].unique():
            method_data = df_results[df_results['method'] == method]
            output_data['average_scores'][method] = {
                'bleu': float(method_data['bleu'].mean()),
                'rouge1': float(method_data['rouge1'].mean()),
                'sbert_cosine': float(method_data['sbert_cosine'].mean()),
                'bertscore_precision': float(method_data['bertscore_precision'].mean()),
                'bertscore_recall': float(method_data['bertscore_recall'].mean()),
                'bertscore_f1': float(method_data['bertscore_f1'].mean())
            }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
else:
    print("No results to save!")