import json
import re
import string
import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
from typing import Any

IOU_THRESH = 0.5
CONTEXT_WINDOW = 100  # characters before/after answer for gold_context_snippet

# Get directory of this script for relative paths
SCRIPT_DIR = Path(__file__).parent

def get_questions_from_csv():
    csv_path = SCRIPT_DIR / "data" / "category_descriptions.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    q_dict = {}
    for i in range(df.shape[0]):
        category = df.iloc[i, 0].split("Category: ")[1]
        description = df.iloc[i, 1].split("Description: ")[1]
        q_dict[category.title()] = description
    return q_dict

qtype_dict = get_questions_from_csv()


def normalize_for_jaccard(text: str) -> str:
    """Normalize text for Jaccard comparison.
    
    Handles common variations:
    - L.L.C. vs LLC
    - Punctuation removal
    - Whitespace normalization
    """
    # Remove common punctuation
    remove_tokens = [".", ",", ";", ":", "'", '"', "(", ")", "-", "_"]
    for token in remove_tokens:
        text = text.replace(token, "")
    
    # Lowercase
    text = text.lower()
    
    # Normalize slashes
    text = text.replace("/", " ")
    
    # Collapse multiple spaces
    text = " ".join(text.split())
    
    return text


def get_jaccard(gt, pred):
    gt = normalize_for_jaccard(gt)
    pred = normalize_for_jaccard(pred)

    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))
    
    # Remove empty strings
    gt_words.discard("")
    pred_words.discard("")

    if not gt_words or not pred_words:
        return 0.0

    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard


def compute_precision_recall(gt_dict, preds_dict, category=None):
    tp, fp, fn = 0, 0, 0

    for key in gt_dict:
        if category and category not in key:
            continue

        substr_ok = "Parties" in key

        answers = gt_dict[key]
        preds = preds_dict[key]

        # first check if answers is empty
        if len(answers) == 0:
            if len(preds) > 0:
                fp += len(preds)  # false positive for each one
        else:
            for ans in answers:
                assert len(ans) > 0
                # check if there is a match
                match_found = False
                for pred in preds:
                    if substr_ok:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                    else:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH
                    if is_match:
                        match_found = True

                if match_found:
                    tp += 1
                else:
                    fn += 1

            # now also get any fps by looping through preds
            for pred in preds:
                # Check if there's a match. if so, don't count (don't want to double count based on the above)
                # but if there's no match, then this is a false positive.
                # (Note: we get the true positives in the above loop instead of this loop so that we don't double count
                # multiple predictions that are matched with the same answer.)
                match_found = False
                for ans in answers:
                    assert len(ans) > 0
                    if substr_ok:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                    else:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH
                    if is_match:
                        match_found = True

                if not match_found:
                    fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    return precision, recall


def process_precisions(precision):
    """
    Processes precisions to ensure that precision and recall don't both get worse
    Assumes the list precision is sorted in order of recalls
    """
    precision_best = precision[::-1]
    for i in range(1, len(precision_best)):
        precision_best[i] = max(precision_best[i-1], precision_best[i])
    precision = precision_best[::-1]
    return precision


def get_aupr(precisions, recalls):
    processed_precisions = process_precisions(precisions)
    aupr = metrics.auc(recalls, processed_precisions)
    if np.isnan(aupr):
        return 0
    return aupr


# ==================== RAG Evaluation Functions ====================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction matches normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def extract_question_type(qa_id: str) -> str:
    """Extract question type from qa_id.
    
    Format: {contract_name}__{question_type}_{index}
    """
    parts = qa_id.split("__")
    if len(parts) >= 2:
        return parts[1].rsplit("_", 1)[0]
    return "Unknown"


def get_gold_context_snippet(context: str, answer_start: int, answer_text: str) -> str:
    """Extract context snippet around the gold answer."""
    start = max(0, answer_start - CONTEXT_WINDOW)
    end = min(len(context), answer_start + len(answer_text) + CONTEXT_WINDOW)
    return context[start:end]


def compute_best_jaccard(prediction: str, gold_answers: list) -> tuple:
    """Compute best Jaccard score against all gold answers.
    
    Returns:
        (best_jaccard_score, is_correct)
    """
    if not prediction or not gold_answers:
        return 0.0, False
    
    best_score = 0.0
    for gold in gold_answers:
        score = get_jaccard(gold, prediction)
        best_score = max(best_score, score)
    
    return best_score, best_score >= IOU_THRESH


def compute_any_exact_match(prediction: str, gold_answers: list) -> bool:
    """Check if prediction exactly matches any gold answer."""
    if not prediction or not gold_answers:
        return False
    
    for gold in gold_answers:
        if is_exact_match(prediction, gold):
            return True
    return False


def evaluate_single_question(
    qa: dict,
    context: str,
    llm_answer: str,
    llm_evidence: str
) -> dict:
    """Evaluate a single question and return result dict for JSONL output."""
    question_id = qa["id"]
    question = qa["question"]
    question_type = extract_question_type(question_id)
    
    # Get gold answers
    gold_answers = [ans["text"] for ans in qa["answers"]]
    
    # Get gold context snippet (from first answer if exists)
    gold_context_snippet = ""
    if qa["answers"]:
        first_ans = qa["answers"][0]
        gold_context_snippet = get_gold_context_snippet(
            context, 
            first_ans["answer_start"], 
            first_ans["text"]
        )
    
    # Compute metrics
    jaccard_score, is_correct = compute_best_jaccard(llm_answer, gold_answers)
    exact_match = compute_any_exact_match(llm_answer, gold_answers)
    
    return {
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "llm_answer": llm_answer,
        "llm_evidence": llm_evidence,
        "gold_answers": gold_answers,
        "gold_context_snippet": gold_context_snippet,
        "jaccard_score": round(jaccard_score, 4),
        "is_exact_match": exact_match,
        "is_correct": is_correct
    }


def load_cuad_dataset(dataset_path: str) -> list:
    """Load CUAD dataset and flatten to list of (qa, context) dicts."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    items = []
    for article in data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                items.append({
                    "qa": qa,
                    "context": context,
                    "title": title
                })
    return items


def compute_rag_summary(results: list) -> dict:
    """Compute summary statistics from all results."""
    total = len(results)
    if total == 0:
        return {"error": "No results"}
    
    correct_count = sum(1 for r in results if r["is_correct"])
    exact_match_count = sum(1 for r in results if r["is_exact_match"])
    
    # Per question type stats
    by_type: dict = {}
    for r in results:
        qtype = r["question_type"]
        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "exact_match": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if r["is_correct"]:
            by_type[qtype]["correct"] += 1
        if r["is_exact_match"]:
            by_type[qtype]["exact_match"] += 1
    
    # Compute accuracy per type
    for qtype in by_type:
        t = by_type[qtype]["total"]
        by_type[qtype]["accuracy"] = round(by_type[qtype]["correct"] / t, 4) if t > 0 else 0
        by_type[qtype]["exact_match_rate"] = round(by_type[qtype]["exact_match"] / t, 4) if t > 0 else 0
    
    return {
        "total_questions": total,
        "correct_count": correct_count,
        "exact_match_count": exact_match_count,
        "accuracy": round(correct_count / total, 4),
        "exact_match_rate": round(exact_match_count / total, 4),
        "by_question_type": by_type
    }


def save_rag_results(results: list, output_dir: str = "."):
    """Save RAG evaluation results to JSONL and summary to JSON."""
    output_path = Path(output_dir)
    
    # Save JSONL (one result per line)
    jsonl_path = output_path / "eval_results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {jsonl_path}")
    
    # Save summary
    summary = compute_rag_summary(results)
    summary_path = output_path / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")
    
    return summary
