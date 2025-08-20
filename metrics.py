from algorithms import user_query, algos, top_n
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr

# -------------------
# Automated Inter-Algorithm Metrics (no ground truth needed)
# -------------------
def compute_overlap_metrics(user_query, algos, top_n=5):
    eval_rows = []

    # Get recommendation lists for each algorithm
    results = {name: func(user_query, top_n)['drug_name'].tolist() for name, func in algos.items()}

    # Compare each pair of algorithms
    for (alg1, res1), (alg2, res2) in combinations(results.items(), 2):
        set1, set2 = set(res1), set(res2)
        intersection = set1 & set2

        precision = len(intersection) / len(set1) if set1 else 0
        recall = len(intersection) / len(set2) if set2 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        jaccard = len(intersection) / len(set1 | set2) if (set1 | set2) else 0

        # Spearman correlation for shared items (if enough overlap)
        if len(intersection) > 1:
            ranks1 = [res1.index(x) for x in intersection]
            ranks2 = [res2.index(x) for x in intersection]
            rho, _ = spearmanr(ranks1, ranks2)
            spearmanr_score = round(rho if rho is not None else 0, 3)
        else:
            spearmanr_score = None

        eval_rows.append({
            "query": user_query,
            "alg_pair": f"{alg1} vs {alg2}",
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "Jaccard": round(jaccard, 3),
            "SpearmanR": spearmanr_score
        })

    return pd.DataFrame(eval_rows)


if __name__ == "__main__":
    metrics_df = compute_overlap_metrics(user_query, algos, top_n)
    print(metrics_df.to_string(index=False))
