"""
Utils package for VFL SHAP Multi-Class Network Intrusion Detection.
Contains model definitions and utility functions.
"""

from .model_utils import (
    LocalEncoder,
    ActiveClassifier,
    VFLModel,
    AgentMetaModel,
    StandardNeuralNetwork
)

from .vfl_utils import (
    simplify_label,
    categorize_feature_by_evidence,
    format_action_readable,
    ATTACK_ACTIONS,
    load_agent_definitions,
    split_features_by_agent_definitions,
    get_evidence_type,
    get_agent_actions_for_attack,
    FIXED_PARTY_NAMES,
    FIXED_AGENT_NAMES,
    get_agent_names
)

__all__ = [
    # Model classes
    'LocalEncoder',
    'ActiveClassifier',
    'VFLModel',
    'AgentMetaModel',
    'StandardNeuralNetwork',
    # Utility functions
    'simplify_label',
    'categorize_feature_by_evidence',
    'format_action_readable',
    'ATTACK_ACTIONS',
    'load_agent_definitions',
    'split_features_by_agent_definitions',
    'get_evidence_type',
    'get_agent_actions_for_attack',
    'FIXED_PARTY_NAMES',
    'FIXED_AGENT_NAMES',
    'get_agent_names',
]
