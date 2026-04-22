import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

def calculate_ece(y_true, y_probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == 1)
            avg_conf_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece

def simulate_profit(y_true, pred_sets, costs):
    """
    Strategy: 
    If set size == 1: auto-decision (if set includes 1 (default), reject)
    Else: manual review
    """
    total_profit = 0
    for i in range(len(y_true)):
        set_size = np.sum(pred_sets[i])
        is_default = y_true[i] == 1
        
        if set_size == 1:
            # Auto-decision: if predicted class 1 (default), reject loan
            # Assuming set={1} means model is confident it's default
            if 1 in np.where(pred_sets[i])[0]:
                total_profit += 0 # Loan rejected
            else:
                # Loan accepted
                if is_default:
                    total_profit += costs["C_FN"]
                else:
                    total_profit += 10000 # Assume some gain from non-default
        else:
            # Manual Review
            total_profit += costs["C_MANUAL"]
            
    return total_profit
