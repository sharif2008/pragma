"""
Utility functions and constants for VFL SHAP Multi-Class Network Intrusion Detection.

This module contains:
- Label simplification functions
- Feature categorization functions
- Attack action mappings
- Party name generation with feature categories
- Action formatting utilities
"""

import json
import re
from pathlib import Path


# ============================================================================
# LABEL SIMPLIFICATION
# ============================================================================

def simplify_label(label_str):
    """
    Group similar attack types into simplified categories.
    
    Args:
        label_str: Original label string from dataset
        
    Returns:
        str: Simplified label category
    """
    label_upper = label_str.upper()
    
    # DDoS (Distributed Denial of Service) -> DDOS
    if 'DDOS' in label_upper:
        return "DDOS"
    
    # All DoS variants (DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris) -> DOS
    if 'DOS' in label_upper and 'DDOS' not in label_upper:
        return "DOS"
    
    # All Web Attack types -> WEBATTACK
    if 'WEB ATTACK' in label_upper or 'WEBATTACK' in label_upper:
        return "WEBATTACK"
    
    # Keep other labels as-is (simplified)
    if 'BENIGN' in label_upper:
        return "BENIGN"
    elif 'BOT' in label_upper:
        return "BOT"
    elif 'PORTSCAN' in label_upper or 'PORT SCAN' in label_upper:
        return "PORTSCAN"
    elif 'FTP' in label_upper and 'PATATOR' in label_upper:
        return "FTPPATATOR"
    elif 'SSH' in label_upper and 'PATATOR' in label_upper:
        return "SSHPATATOR"
    elif 'HEARTBLEED' in label_upper:
        return "HEARTBLEED"
    elif 'INFILTRATION' in label_upper:
        return "INFILTRATION"
    else:
        # Default: take first word or first 8 chars
        words = label_str.split()
        if len(words) > 0:
            return words[0].upper()[:8]
        return label_str.upper()[:8]


# ============================================================================
# FIXED PARTY NAMES
# ============================================================================

# Fixed static party names (used consistently across all functions)
FIXED_PARTY_NAMES = {
    1: "evidence_volume_rate_agent1",
    2: "evidence_packet_size_agent2",
    3: "evidence_timing_direction_agent3"
}

# Fixed agent names as a list (for easy indexing: agent_names[0], agent_names[1], agent_names[2])
FIXED_AGENT_NAMES = [
    FIXED_PARTY_NAMES[1],  # Agent 1 (index 0)
    FIXED_PARTY_NAMES[2],  # Agent 2 (index 1)
    FIXED_PARTY_NAMES[3]   # Agent 3 (index 2)
]

def get_agent_names():
    """
    Get fixed agent names as a list.
    
    Returns:
        list: List of 3 fixed agent names in order [agent1, agent2, agent3]
    """
    return FIXED_AGENT_NAMES.copy()


# ============================================================================
# ATTACK ACTION MAPPINGS
# ============================================================================

ATTACK_ACTIONS = {
    'BENIGN': " No action, log only",
    'DDOS': "rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers",
    'DOS': "rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers",
    'SSHPATATOR': "brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation",
    'FTPPATATOR': "brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation",
    'PORTSCAN': "scan detection: block scanner IP, tarpitting, tighten firewall rules",
    'WEBATTACK': "WAF rules, block patterns, patching, isolate vulnerable service",
    'BOT': "bot detection: block bot IPs, implement CAPTCHA, rate limiting",
    'OTHERS': "safe response: alert + collect evidence + temporary throttling (not hard block)"
}


# ============================================================================
# AGENT DEFINITIONS FROM JSON (agentic_features.json)
# ============================================================================


def _normalize_feature_name(name):
    """Normalize for matching: lowercase, spaces to underscores."""
    if not isinstance(name, str):
        return name
    return name.lower().strip().replace(" ", "_")


def load_agent_definitions(json_path):
    """
    Load agent definitions from agentic_features.json (schema_version 1.0).
    Expects: {"agents": {"RAN": {"description", "logged_features", "action_capabilities"}, ...}}.

    Returns:
        dict: {
            "agent_names": ["RAN", "Edge", "Core"],
            "agent_features": [list, list, list],  # 3 lists of feature names (logged_features)
            "agent_domains": list of description strings,
            "agent_actions": [list, list, list],   # action_capabilities per agent
            "feature_categories": dict              # for compatibility
        }
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent definitions file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    agents = raw.get("agents") or {}
    agent_order = ["RAN", "Edge", "Core"]
    agent_names = [n for n in agent_order if n in agents]
    agent_features = []
    agent_domains = []
    agent_actions = []
    for name in agent_order:
        a = agents.get(name) or {}
        agent_features.append(list(a.get("logged_features") or []))
        agent_domains.append(a.get("description") or name)
        agent_actions.append(list(a.get("action_capabilities") or []))

    feature_categories = {
        "evidence_volume_rate": agent_features[0] if len(agent_features) > 0 else [],
        "evidence_packet_size": agent_features[1] if len(agent_features) > 1 else [],
        "evidence_timing_direction": agent_features[2] if len(agent_features) > 2 else [],
    }
    return {
        "agent_names": agent_names,
        "agent_features": agent_features,
        "agent_domains": agent_domains,
        "agent_actions": agent_actions,
        "feature_categories": feature_categories,
    }


def load_attack_options(json_path):
    """
    Load attack options from attack_options.json (schema_version 1.0).
    Expects: {"attacks": {"BENIGN": ["log only", ...], "DDOS": [...], ...}}.
    Not used for splitting, training, or inference; for reference/display only.

    Returns:
        dict: {"schema_version": ..., "attacks": {attack_type: [action_strings]}}
    """
    path = Path(json_path)
    if not path.exists():
        return {"schema_version": "1.0", "attacks": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def split_features_by_agent_definitions(dataset_columns, agent_definitions):
    """
    Split dataset columns into three agent feature lists using the loaded agent definitions.
    Only features that exist in dataset_columns are included; order follows the JSON definition.
    Matching is exact first, then normalized (lowercase, spaces to underscores) so JSON names
    like "flow_id" can match dataset columns like "Flow ID".

    Args:
        dataset_columns: List of column names in the dataset (e.g. df.columns or all_features).
        agent_definitions: dict from load_agent_definitions().

    Returns:
        tuple: (agent1_features, agent2_features, agent3_features, feature_categories)
        - feature_categories is the same structure as in agent_definitions for compatibility.
    """
    agent_features = agent_definitions["agent_features"]
    feature_categories = agent_definitions.get("feature_categories", {})
    set_cols = set(dataset_columns)
    # Normalized (lower, spaces->underscores) -> original column name for flexible matching
    norm_to_col = {_normalize_feature_name(c): c for c in dataset_columns}

    def resolve(f):
        if f in set_cols:
            return f
        n = _normalize_feature_name(f)
        return norm_to_col.get(n)

    out = []
    for feats in agent_features:
        # Keep only features present in the dataset, preserve order from JSON
        filtered = []
        seen = set()
        for f in feats:
            c = resolve(f)
            if c is not None and c not in seen:
                filtered.append(c)
                seen.add(c)
        out.append(filtered)

    # Ensure we have exactly 3 agents
    while len(out) < 3:
        out.append([])
    agent1_features, agent2_features, agent3_features = out[0], out[1], out[2]

    # Update feature_categories to reflect filtered lists (for reporting)
    feature_categories = {
        "evidence_volume_rate": agent1_features,
        "evidence_packet_size": agent2_features,
        "evidence_timing_direction": agent3_features,
    }
    return agent1_features, agent2_features, agent3_features, feature_categories


# ============================================================================
# FEATURE CATEGORIZATION
# ============================================================================

def categorize_feature_by_evidence(feature_name):
    """
    Categorize features by evidence type for IDS-style partitioning.
    Each agent represents a different sensor/evidence type.
    
    Strategy:
    - Agent 1: Volume/Rate (DoS/DDoS) - flow duration, packet/byte counts, rates
    - Agent 2: Packet Size (Scans/Web) - packet length stats, size distributions
    - Agent 3: Timing/Direction (Brute Force/Scans) - IAT, intervals, directionality
    
    Args:
        feature_name: Name of the feature to categorize
        
    Returns:
        str: Evidence type category ('evidence_volume_rate', 'evidence_packet_size', 
             or 'evidence_timing_direction')
    """
    feat_lower = feature_name.lower()
    
    # Agent 1: Volume/Rate evidence (DoS/DDoS strong)
    # Priority: flow duration, total packets/bytes, packet/byte counts, rates
    if any(x in feat_lower for x in ['flow_duration', 'duration_ms', 'total_fwd_packets', 
                                     'total_backward_packets', 'total_length_of_fwd_packets',
                                     'total_length_of_bwd_packets', 'flow_bytes', 'flow_packets',
                                     'packet_count', 'byte_count', 'total_packets', 'total_bytes']):
        return 'evidence_volume_rate'
    
    # Also volume/rate: any feature with "total" and (packet/byte/flow) but NOT length/size stats
    if 'total' in feat_lower and any(x in feat_lower for x in ['packet', 'byte', 'flow', 'fwd', 'bwd']):
        if not any(x in feat_lower for x in ['length', 'size', 'mean', 'std', 'max', 'min', 'avg']):
            return 'evidence_volume_rate'
    
    # Agent 2: Packet size distribution (scans + web attacks)
    # Priority: packet length stats (mean/std/max/min), size distributions, header lengths
    if any(x in feat_lower for x in ['packet_length', 'packet_size', 'ps_', 'fwd_packet_length',
                                     'bwd_packet_length', 'header_length', 'segment_size',
                                     'avg_packet', 'packet_length_mean', 'packet_length_std',
                                     'packet_length_max', 'packet_length_min']):
        return 'evidence_packet_size'
    
    # Also size: any feature with length/size AND statistical measure
    if any(x in feat_lower for x in ['length', 'size']) and any(x in feat_lower for x in ['mean', 'std', 'max', 'min', 'avg', 'var', 'stddev']):
        return 'evidence_packet_size'
    
    # Agent 3: Timing/Directionality/Burstiness (brute force + scanning)
    # Priority: IAT, PIAT, timing intervals, active/idle, directionality ratios, burst patterns
    if any(x in feat_lower for x in ['piat', 'iat', 'interval', 'active_mean', 'active_std',
                                     'idle_mean', 'idle_std', 'active_max', 'idle_max',
                                     'src2dst', 'dst2src', 'forward', 'backward', 'fwd_', 'bwd_',
                                     'down_up_ratio', 'burst', 'subflow']):
        return 'evidence_timing_direction'
    
    # Also timing: any feature with time/timing/interval
    if any(x in feat_lower for x in ['time', 'timing', 'interval']):
        return 'evidence_timing_direction'
    
    # Protocol/Flag features: assign to Agent 3 (timing/direction)
    if any(x in feat_lower for x in ['syn', 'ack', 'fin', 'rst', 'urg', 'cwr', 'ece', 'psh',
                                     'flag', 'protocol', 'port', 'udp', 'tcp', 'init_win']):
        return 'evidence_timing_direction'
    
    # Fallback: check for volume indicators
    if any(x in feat_lower for x in ['total', 'sum', 'count']) and 'length' not in feat_lower:
        return 'evidence_volume_rate'
    
    # Default: assign to timing/direction (most diverse category)
    return 'evidence_timing_direction'


def get_evidence_type(features):
    """
    Determine evidence type for party based on features.
    
    Args:
        features: List of feature names
        
    Returns:
        str: Dominant evidence type
    """
    evidence_counts = {
        'evidence_volume_rate': 0,
        'evidence_packet_size': 0,
        'evidence_timing_direction': 0
    }
    
    for feat in features:
        cat = categorize_feature_by_evidence(feat)
        if cat in evidence_counts:
            evidence_counts[cat] += 1
    
    # Return dominant evidence type
    return max(evidence_counts.items(), key=lambda x: x[1])[0]


def get_feature_category_summary(features):
    """
    Get a summary of feature categories for a party.
    
    Args:
        features: List of feature names
        
    Returns:
        dict: Summary with category counts and top feature types
    """
    category_counts = {
        'volume_rate': 0,
        'packet_size': 0,
        'timing_direction': 0,
        'protocol': 0,
        'other': 0
    }
    
    feature_types = {
        'duration': 0,
        'packet_count': 0,
        'byte_count': 0,
        'packet_size_stats': 0,
        'timing_intervals': 0,
        'direction': 0,
        'protocol_flags': 0
    }
    
    for feat in features:
        feat_lower = feat.lower()
        cat = categorize_feature_by_evidence(feat)
        
        if cat == 'evidence_volume_rate':
            category_counts['volume_rate'] += 1
        elif cat == 'evidence_packet_size':
            category_counts['packet_size'] += 1
        elif cat == 'evidence_timing_direction':
            category_counts['timing_direction'] += 1
        
        # Count specific feature types
        if 'duration' in feat_lower:
            feature_types['duration'] += 1
        if 'packet_count' in feat_lower or 'packet_num' in feat_lower:
            feature_types['packet_count'] += 1
        if 'byte' in feat_lower and 'count' in feat_lower:
            feature_types['byte_count'] += 1
        if any(x in feat_lower for x in ['ps_', 'packet_length', 'packet_size']):
            feature_types['packet_size_stats'] += 1
        if any(x in feat_lower for x in ['piat', 'iat', 'interval']):
            feature_types['timing_intervals'] += 1
        if any(x in feat_lower for x in ['src2dst', 'dst2src', 'fwd', 'bwd', 'bidirectional']):
            feature_types['direction'] += 1
        if any(x in feat_lower for x in ['syn', 'ack', 'fin', 'rst', 'urg', 'cwr', 'ece', 'psh', 'port', 'protocol']):
            feature_types['protocol_flags'] += 1
    
    return {
        'category_counts': category_counts,
        'feature_types': feature_types,
        'total_features': len(features)
    }


# ============================================================================
# PARTY NAME GENERATION (UNIQUE WITH FEATURE CATEGORY)
# ============================================================================

def generate_party_name(features, party_num):
    """
    Generate fixed party name based on party number.
    
    Fixed party naming convention:
    1) evidence_volume_rate_agent1 (Party 1)
    2) evidence_packet_size_agent2 (Party 2)
    3) evidence_timing_direction_agent3 (Party 3)
    
    Args:
        features: List of feature names (kept for compatibility, not used)
        party_num: Party number (1, 2, or 3)
        
    Returns:
        str: Fixed party name based on party number
    """
    if party_num not in FIXED_PARTY_NAMES:
        raise ValueError(f"Invalid party number: {party_num}. Must be 1, 2, or 3.")
    
    return FIXED_PARTY_NAMES[party_num]


def generate_domain(features, party_num):
    """
    Generate party domain description based on evidence type and feature category.
    
    Args:
        features: List of feature names
        party_num: Party number (1, 2, or 3)
        
    Returns:
        str: Domain description with feature category details
    """
    evidence_type = get_evidence_type(features)
    category_summary = get_feature_category_summary(features)
    
    if evidence_type == 'evidence_volume_rate':
        domain = f"Volume & Rate Analysis (DoS/DDoS Detection) - {category_summary['total_features']} features"
        if category_summary['feature_types']['duration'] > 0:
            domain += " | Focus: Flow Duration & Packet/Byte Counts"
        else:
            domain += " | Focus: Packet & Byte Volume Metrics"
    
    elif evidence_type == 'evidence_packet_size':
        domain = f"Packet Size Distribution Analysis (Scan/Web Attack Detection) - {category_summary['total_features']} features"
        if category_summary['feature_types']['packet_size_stats'] > 0:
            domain += " | Focus: Packet Length Statistics"
        else:
            domain += " | Focus: Size Distribution Patterns"
    
    else:  # evidence_timing_direction
        domain = f"Timing & Directionality Analysis (Brute Force/Scan Detection) - {category_summary['total_features']} features"
        if category_summary['feature_types']['protocol_flags'] > 0:
            domain += " | Focus: Protocol Flags & Port Analysis"
        elif category_summary['feature_types']['direction'] > 0:
            domain += " | Focus: Flow Direction & Bidirectional Patterns"
        else:
            domain += " | Focus: Inter-Arrival Time & Timing Intervals"
    
    return domain


# ============================================================================
# ACTION GENERATION
# ============================================================================

def generate_action(features, party_num):
    """
    Generate attack-type specific actions based on evidence type.
    Each party has different actions for different attack types.
    
    Args:
        features: List of feature names
        party_num: Party number (1, 2, or 3)
        
    Returns:
        str: Formatted action string
    """
    evidence_type = get_evidence_type(features)
    
    # Map evidence type to which attacks it detects best (primary focus)
    evidence_to_primary_attacks = {
        'evidence_volume_rate': ['DDOS', 'DOS'],
        'evidence_packet_size': ['PORTSCAN', 'WEBATTACK'],
        'evidence_timing_direction': ['SSHPATATOR', 'FTPPATATOR', 'PORTSCAN']
    }
    
    primary_attacks = evidence_to_primary_attacks.get(evidence_type, [])
    
    # Build comprehensive action mapping for all attack types
    action_dict = {}
    for attack_type in ATTACK_ACTIONS.keys():
        if attack_type in primary_attacks:
            # Primary detection - use full action
            action_dict[attack_type] = ATTACK_ACTIONS[attack_type]
        else:
            # Secondary detection - use generic monitoring
            if evidence_type == 'evidence_volume_rate':
                action_dict[attack_type] = "monitor volume/rate patterns and alert"
            elif evidence_type == 'evidence_packet_size':
                action_dict[attack_type] = "monitor packet size patterns and alert"
            elif evidence_type == 'evidence_timing_direction':
                action_dict[attack_type] = "monitor timing/direction patterns and alert"
            else:
                action_dict[attack_type] = "monitor and alert"
    
    # Return formatted string for display
    primary_str = "; ".join([f"{atk}: {action_dict[atk]}" for atk in primary_attacks])
    return primary_str if primary_str else "monitor and alert"


def get_party_actions_for_attack(party_features, attack_type, evidence_type=None):
    """
    Get action for specific party and attack type.
    
    Args:
        party_features: List of features for this party
        attack_type: Attack type name (e.g., 'DDOS', 'PORTSCAN')
        evidence_type: Optional evidence type (if None, will be determined from features)
        
    Returns:
        str: Recommended action for this party+attack combination
    """
    if evidence_type is None:
        evidence_type = get_evidence_type(party_features)
    
    # Primary attacks per evidence type
    evidence_to_primary = {
        'evidence_volume_rate': ['DDOS', 'DOS'],
        'evidence_packet_size': ['PORTSCAN', 'WEBATTACK'],
        'evidence_timing_direction': ['SSHPATATOR', 'FTPPATATOR', 'PORTSCAN']
    }
    
    primary_attacks = evidence_to_primary.get(evidence_type, [])
    
    if attack_type.upper() in primary_attacks:
        return ATTACK_ACTIONS.get(attack_type.upper(), "monitor and alert")
    else:
        # Secondary detection
        if evidence_type == 'evidence_volume_rate':
            return "monitor volume/rate patterns and alert"
        elif evidence_type == 'evidence_packet_size':
            return "monitor packet size patterns and alert"
        elif evidence_type == 'evidence_timing_direction':
            return "monitor timing/direction patterns and alert"
        return "monitor and alert"


# ============================================================================
# ACTION FORMATTING
# ============================================================================

def format_action_readable(action_string):
    """
    Convert action string to human-readable bullet points.
    
    Args:
        action_string: Comma-separated string of actions
        
    Returns:
        str: Formatted string with bullet points and descriptions
    """
    actions = [a.strip() for a in action_string.split(',')]
    action_descriptions = {
        "rate-limit": "Rate Limiting: Limit incoming request rate per IP/connection",
        "SYN cookies": "SYN Cookies: Enable SYN flood protection",
        "WAF rules": "WAF Rules: Apply web application firewall rules",
        "drop bursts": "Drop Bursts: Immediately drop traffic bursts",
        "auto-scale": "Auto-Scale: Scale up resources to handle load",
        "block top talkers": "Block Top Talkers: Block IPs with highest traffic volume",
        "fail2ban-style blocking": "Fail2Ban: Automatically ban IPs after failed login attempts",
        "lockout": "Account Lockout: Lock accounts after multiple failed attempts",
        "MFA": "Multi-Factor Authentication: Require additional authentication",
        "geo/IP reputation": "Geo/IP Reputation: Block based on geographic location or IP reputation",
        "block scanner IP": "Block Scanner IP: Immediately block the scanning IP address",
        "tarpitting": "Tarpitting: Slow down scanner responses to waste attacker time",
        "tighten firewall rules": "Firewall Rules: Tighten firewall rules to restrict access",
        "block patterns": "Block Patterns: Block known attack patterns",
        "patching": "Patching: Apply security patches to vulnerable services",
        "isolate vulnerable service": "Isolate Service: Isolate the vulnerable service from network",
        "block bot IPs": "Block Bot IPs: Block known bot IP addresses",
        "CAPTCHA": "CAPTCHA: Implement CAPTCHA challenges",
        "rate limiting": "Rate Limiting: Limit requests from suspected bots"
    }
    
    formatted = []
    for action in actions:
        action_lower = action.lower()
        found = False
        for key, desc in action_descriptions.items():
            if key in action_lower:
                formatted.append(f"     • {desc}")
                found = True
                break
        if not found:
            formatted.append(f"     • {action.strip()}")
    
    return "\n".join(formatted)


def get_evidence_type(features):
    """
    Determine evidence type for agent based on features.
    
    Args:
        features: List of feature names
        
    Returns:
        str: Dominant evidence type
    """
    # Check which evidence category dominates
    evidence_counts = {
        'evidence_volume_rate': 0,
        'evidence_packet_size': 0,
        'evidence_timing_direction': 0
    }
    
    for feat in features:
        cat = categorize_feature_by_evidence(feat)
        if cat in evidence_counts:
            evidence_counts[cat] += 1
    
    # Return dominant evidence type
    dominant = max(evidence_counts.items(), key=lambda x: x[1])[0]
    return dominant


def generate_agent_name(features, agent_num):
    """
    Generate fixed agent name based on agent number.
    
    Fixed agent naming convention (matches party names):
    1) evidence_volume_rate_agent1 (Agent 1)
    2) evidence_packet_size_agent2 (Agent 2)
    3) evidence_timing_direction_agent3 (Agent 3)
    
    Args:
        features: List of feature names (kept for compatibility, not used)
        agent_num: Agent number (1, 2, or 3)
        
    Returns:
        str: Fixed agent name based on agent number
    """
    if agent_num not in FIXED_PARTY_NAMES:
        raise ValueError(f"Invalid agent number: {agent_num}. Must be 1, 2, or 3.")
    
    return FIXED_PARTY_NAMES[agent_num]


def generate_domain(features, agent_num):
    """
    Generate agent domain description based on evidence type.
    
    Args:
        features: List of feature names
        agent_num: Agent number (1, 2, or 3)
        
    Returns:
        str: Domain description
    """
    evidence_type = get_evidence_type(features)
    
    domain_map = {
        'evidence_volume_rate': "Volume & Rate Analysis (DoS/DDoS Detection)",
        'evidence_packet_size': "Packet Size Distribution Analysis (Scan/Web Attack Detection)",
        'evidence_timing_direction': "Timing & Directionality Analysis (Brute Force/Scan Detection)"
    }
    return domain_map.get(evidence_type, f"Network Analysis Agent {agent_num}")


def generate_action(features, agent_num):
    """
    Generate attack-type specific actions based on evidence type.
    Each agent has different actions for different attack types.
    
    Args:
        features: List of feature names
        agent_num: Agent number (1, 2, or 3)
        
    Returns:
        str: Formatted action string
    """
    evidence_type = get_evidence_type(features)
    
    # Map evidence type to which attacks it detects best (primary focus)
    evidence_to_primary_attacks = {
        'evidence_volume_rate': ['DDOS', 'DOS'],  # Agent 1 detects DoS/DDoS
        'evidence_packet_size': ['PORTSCAN', 'WEBATTACK'],  # Agent 2 detects scans/web
        'evidence_timing_direction': ['SSHPATATOR', 'FTPPATATOR', 'PORTSCAN']  # Agent 3 detects brute force/scans
    }
    
    primary_attacks = evidence_to_primary_attacks.get(evidence_type, [])
    
    # Build comprehensive action mapping for all attack types
    action_dict = {}
    for attack_type in ATTACK_ACTIONS.keys():
        if attack_type in primary_attacks:
            # Primary detection - use full action
            action_dict[attack_type] = ATTACK_ACTIONS[attack_type]
        else:
            # Secondary detection - use generic monitoring
            if evidence_type == 'evidence_volume_rate':
                action_dict[attack_type] = "monitor volume/rate patterns and alert"
            elif evidence_type == 'evidence_packet_size':
                action_dict[attack_type] = "monitor packet size patterns and alert"
            elif evidence_type == 'evidence_timing_direction':
                action_dict[attack_type] = "monitor timing/direction patterns and alert"
            else:
                action_dict[attack_type] = "monitor and alert"
    
    # Return formatted string for display
    primary_str = "; ".join([f"{atk}: {action_dict[atk]}" for atk in primary_attacks])
    return primary_str if primary_str else "monitor and alert"

def get_feature_semantic_group(feature_name):
    """
    Group features by semantic similarity to keep related features together.
    This helps maintain meaningful feature clusters within each agent.
    
    Returns:
        str: Semantic group name
    """
    feat_lower = feature_name.lower()
    
    # Group 1: Forward direction features
    if any(x in feat_lower for x in ['fwd_', 'forward', 'src2dst', 'down_up']):
        return 'forward_direction'
    
    # Group 2: Backward direction features
    if any(x in feat_lower for x in ['bwd_', 'backward', 'dst2src']):
        return 'backward_direction'
    
    # Group 3: Bidirectional features
    if 'bidirectional' in feat_lower or 'bi_' in feat_lower:
        return 'bidirectional'
    
    # Group 4: Packet count/volume features
    if any(x in feat_lower for x in ['packet_count', 'packet_num', 'total_packet', 'flow_packet']):
        return 'packet_volume'
    
    # Group 5: Byte/flow size features
    if any(x in feat_lower for x in ['byte', 'flow_bytes', 'total_length', 'total_size']):
        return 'byte_volume'
    
    # Group 6: Duration/timing features
    if any(x in feat_lower for x in ['duration', 'time', 'interval', 'iat', 'piat']):
        return 'timing'
    
    # Group 7: Packet size statistics
    if any(x in feat_lower for x in ['packet_length', 'packet_size', 'ps_', 'header_length']):
        return 'packet_size_stats'
    
    # Group 8: Protocol/flag features
    if any(x in feat_lower for x in ['syn', 'ack', 'fin', 'rst', 'urg', 'cwr', 'ece', 'psh', 'flag', 'protocol']):
        return 'protocol_flags'
    
    # Group 9: Rate/throughput features
    if any(x in feat_lower for x in ['rate', 'throughput', 'bps', 'pps']):
        return 'rate'
    
    # Group 10: Statistical features (mean, std, max, min)
    if any(x in feat_lower for x in ['_mean', '_std', '_max', '_min', '_avg', '_var']):
        return 'statistics'
    
    # Default: other
    return 'other'


def split_features_balanced(all_features, num_agents=3, min_features_per_agent=20, balance_threshold=0.15, random_seed=42):
    """
    Split features into agents in a balanced way while preserving semantic grouping.
    
    Strategy:
    1. First categorize features by evidence type (for attack detection relevance)
    2. Group features by semantic similarity within each category
    3. Ensure minimum features per agent (default: 20)
    4. Redistribute to balance while keeping semantically similar features together
    
    Args:
        all_features: List of all feature names
        num_agents: Number of agents (default: 3)
        min_features_per_agent: Minimum number of features per agent (default: 20)
        balance_threshold: Maximum allowed difference ratio between largest and smallest agent (default: 0.15 = 15%)
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (agent1_features, agent2_features, agent3_features, feature_categories)
    """
    import numpy as np
    import random
    from collections import defaultdict
    
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Step 1: Categorize features by evidence type (for attack relevance)
    feature_categories = {
        'evidence_volume_rate': [],      # Agent 1: Volume/Rate (DoS/DDoS)
        'evidence_packet_size': [],      # Agent 2: Packet Size (Scans/Web)
        'evidence_timing_direction': []  # Agent 3: Timing/Direction (Brute Force/Scans)
    }
    
    for feat in all_features:
        cat = categorize_feature_by_evidence(feat)
        if cat in feature_categories:
            feature_categories[cat].append(feat)
        else:
            # Default assignment if categorization fails
            feature_categories['evidence_timing_direction'].append(feat)
    
    # Step 2: Group features by semantic similarity within each category
    semantic_groups = defaultdict(lambda: defaultdict(list))
    for cat, feats in feature_categories.items():
        for feat in feats:
            sem_group = get_feature_semantic_group(feat)
            semantic_groups[cat][sem_group].append(feat)
    
    # Step 3: Check if we need to redistribute to meet minimum requirements
    category_sizes = {cat: len(feats) for cat, feats in feature_categories.items()}
    total_features = sum(category_sizes.values())
    target_size_per_agent = max(total_features / num_agents, min_features_per_agent)
    
    # Step 4: Redistribute features to ensure minimum per agent and balance
    # Strategy: Move entire semantic groups to maintain feature coherence
    sorted_categories = sorted(category_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # If any category has less than minimum, we need to redistribute
    for cat, size in category_sizes.items():
        if size < min_features_per_agent:
            # Need to get features from larger categories
            needed = min_features_per_agent - size
            
            for larger_cat, larger_size in sorted_categories:
                if larger_cat == cat or larger_size <= min_features_per_agent:
                    continue
                
                available_groups = list(semantic_groups[larger_cat].items())
                available_groups.sort(key=lambda x: len(x[1]))
                
                for sem_group_name, group_feats in available_groups:
                    if needed <= 0:
                        break
                    
                    # Move entire semantic group if it fits
                    if len(group_feats) <= needed or len(group_feats) <= (larger_size - min_features_per_agent):
                        # Move the entire group
                        semantic_groups[cat][sem_group_name].extend(group_feats)
                        del semantic_groups[larger_cat][sem_group_name]
                        needed -= len(group_feats)
                        category_sizes[cat] += len(group_feats)
                        category_sizes[larger_cat] -= len(group_feats)
    
    # Step 5: Further balance by moving semantic groups if still imbalanced
    # Calculate target size
    target_size = max(total_features / num_agents, min_features_per_agent)
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        category_sizes = {cat: sum(len(feats) for feats in groups.values()) 
                          for cat, groups in semantic_groups.items()}
        sizes = list(category_sizes.values())
        max_size = max(sizes)
        min_size = min(sizes)
        
        # Check if balanced enough and meets minimum
        if min_size >= min_features_per_agent and (max_size - min_size) <= target_size * balance_threshold:
            break
        
        largest_cat = max(category_sizes.items(), key=lambda x: x[1])[0]
        smallest_cat = min(category_sizes.items(), key=lambda x: x[1])[0]
        
        if largest_cat == smallest_cat:
            break
        
        available_groups = list(semantic_groups[largest_cat].items())
        if not available_groups:
            break
        
        available_groups.sort(key=lambda x: len(x[1]))
        
        moved = False
        for sem_group_name, group_feats in available_groups:
            # Only move if it doesn't violate minimum for source category
            if (category_sizes[largest_cat] - len(group_feats) >= min_features_per_agent and
                len(group_feats) <= (max_size - min_size) // 2):
                semantic_groups[smallest_cat][sem_group_name].extend(group_feats)
                del semantic_groups[largest_cat][sem_group_name]
                moved = True
                break
        
        if not moved:
            break
        
        iteration += 1
    
    # Step 6: Reconstruct feature lists from semantic groups
    agent1_features = []
    agent2_features = []
    agent3_features = []
    
    for sem_group_name, group_feats in semantic_groups['evidence_volume_rate'].items():
        agent1_features.extend(group_feats)
    
    for sem_group_name, group_feats in semantic_groups['evidence_packet_size'].items():
        agent2_features.extend(group_feats)
    
    for sem_group_name, group_feats in semantic_groups['evidence_timing_direction'].items():
        agent3_features.extend(group_feats)
    
    # Step 7: Final check - ensure minimum features per agent
    all_agent_features = [agent1_features, agent2_features, agent3_features]
    agent_sizes = [len(p) for p in all_agent_features]
    
    # If any agent has less than minimum, redistribute from others
    for i, size in enumerate(agent_sizes):
        if size < min_features_per_agent:
            needed = min_features_per_agent - size
            # Get from largest agent
            largest_idx = agent_sizes.index(max(agent_sizes))
            if largest_idx != i and agent_sizes[largest_idx] > min_features_per_agent:
                # Move features (prefer keeping semantic groups together)
                to_move = min(needed, agent_sizes[largest_idx] - min_features_per_agent)
                moved_feats = all_agent_features[largest_idx][:to_move]
                all_agent_features[largest_idx] = all_agent_features[largest_idx][to_move:]
                all_agent_features[i].extend(moved_feats)
                agent_sizes[i] += to_move
                agent_sizes[largest_idx] -= to_move
    
    # Step 8: Shuffle features within each agent (but semantic groups are already together)
    for agent_feats in all_agent_features:
        np.random.shuffle(agent_feats)
    
    # Reconstruct feature_categories for return
    feature_categories = {
        'evidence_volume_rate': agent1_features,
        'evidence_packet_size': agent2_features,
        'evidence_timing_direction': agent3_features
    }
    
    return all_agent_features[0], all_agent_features[1], all_agent_features[2], feature_categories


def get_agent_actions_for_attack(agent_features, attack_type, evidence_type=None):
    """
    Get action for specific agent and attack type.
    
    Args:
        agent_features: List of features for this agent
        attack_type: Attack type name (e.g., 'DDOS', 'PORTSCAN')
        evidence_type: Optional evidence type (if None, will be determined from features)
        
    Returns:
        str: Recommended action for this agent+attack combination
    """
    if evidence_type is None:
        evidence_type = get_evidence_type(agent_features)
    
    # Primary attacks per evidence type
    evidence_to_primary = {
        'evidence_volume_rate': ['DDOS', 'DOS'],
        'evidence_packet_size': ['PORTSCAN', 'WEBATTACK'],
        'evidence_timing_direction': ['SSHPATATOR', 'FTPPATATOR', 'PORTSCAN']
    }
    
    primary_attacks = evidence_to_primary.get(evidence_type, [])
    
    if attack_type.upper() in primary_attacks:
        return ATTACK_ACTIONS.get(attack_type.upper(), "monitor and alert")
    else:
        # Secondary detection
        if evidence_type == 'evidence_volume_rate':
            return "monitor volume/rate patterns and alert"
        elif evidence_type == 'evidence_packet_size':
            return "monitor packet size patterns and alert"
        elif evidence_type == 'evidence_timing_direction':
            return "monitor timing/direction patterns and alert"
        return "monitor and alert"
