"""
injecagent_benchmark.py
Runs your 3 detectors against real test cases adapted from the
InjecAgent benchmark (uiuc-kang-lab/InjecAgent), instead of only
your own synthetic attacks. This gives a credible, external accuracy
number to report.
"""

import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from rule_detector import detect_injection as rule_detect
from intent_drift_detector import detect_intent_drift

# ---------------------------------------------------------
# 1. Load InjecAgent data
# ---------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

user_cases = load_jsonl("InjecAgent/data/user_cases.jsonl")
attacker_cases_dh = load_jsonl("InjecAgent/data/attacker_cases_dh.jsonl")

print(f"Loaded {len(user_cases)} user cases, {len(attacker_cases_dh)} attacker cases (direct harm).")

# ---------------------------------------------------------
# 2. Build test set: (intent, tool_output, is_attack)
# ---------------------------------------------------------

random.seed(42)
NUM_SAMPLES = 60  # keep it manageable for a quick run

test_set = []

# Poisoned cases: combine a user case's template with a random attacker instruction
for uc in random.sample(user_cases, min(NUM_SAMPLES // 2, len(user_cases))):
    if "<Attacker Instruction>" not in uc["Tool Response Template"]:
        continue
    attacker = random.choice(attacker_cases_dh)
    poisoned_output = uc["Tool Response Template"].replace(
        "<Attacker Instruction>", attacker["Attacker Instruction"]
    )
    test_set.append({
        "intent": uc["User Instruction"],
        "text": poisoned_output,
        "is_attack": True,
    })

# Clean cases: same templates, but with a generic benign filler instead of an attack
BENIGN_FILLERS = [
    "Everything looks good, no issues found.",
    "This item is in stock and ready to ship.",
    "The content was reviewed and approved.",
    "No further action is required at this time.",
]
for uc in random.sample(user_cases, min(NUM_SAMPLES // 2, len(user_cases))):
    if "<Attacker Instruction>" not in uc["Tool Response Template"]:
        continue
    clean_output = uc["Tool Response Template"].replace(
        "<Attacker Instruction>", random.choice(BENIGN_FILLERS)
    )
    test_set.append({
        "intent": uc["User Instruction"],
        "text": clean_output,
        "is_attack": False,
    })

print(f"Built benchmark test set with {len(test_set)} cases "
      f"({sum(t['is_attack'] for t in test_set)} attack, "
      f"{sum(not t['is_attack'] for t in test_set)} clean).")

# ---------------------------------------------------------
# 3. Train the ML classifier on your existing dataset.csv
# ---------------------------------------------------------

print("\nTraining ML classifier on dataset.csv...")
df = pd.read_csv("dataset.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(df["text"].tolist())
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)
print("ML classifier ready.\n")

# ---------------------------------------------------------
# 4. Run all 3 detectors on the InjecAgent-based test set
# ---------------------------------------------------------

def run_rule(text):
    return rule_detect(text)["flagged"]

def run_ml(text):
    emb = embedder.encode([text])
    return bool(clf.predict(emb)[0] == 1)

def run_drift(intent, text):
    return detect_intent_drift(intent, text, threshold=0.25)["flagged"]

results = []
for case in test_set:
    results.append({
        "is_attack": case["is_attack"],
        "rule_based": run_rule(case["text"]),
        "ml_classifier": run_ml(case["text"]),
        "intent_drift": run_drift(case["intent"], case["text"]),
    })

results_df = pd.DataFrame(results)

print("=== INJECAGENT-BASED BENCHMARK RESULTS ===")
for detector in ["rule_based", "ml_classifier", "intent_drift"]:
    correct = (results_df[detector] == results_df["is_attack"]).sum()
    total = len(results_df)
    tp = ((results_df[detector] == True) & (results_df["is_attack"] == True)).sum()
    fp = ((results_df[detector] == True) & (results_df["is_attack"] == False)).sum()
    fn = ((results_df[detector] == False) & (results_df["is_attack"] == True)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"{detector:15s}: accuracy={correct}/{total} ({correct/total*100:.1f}%) "
          f"| precision={precision:.2f} | recall={recall:.2f}")

results_df.to_csv("injecagent_results.csv", index=False)
print("\nSaved to injecagent_results.csv")

