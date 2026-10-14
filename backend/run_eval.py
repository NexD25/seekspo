"""
run_eval.py -- evaluate zero-shot vs few-shot sentiment/aspect extraction
against the hand-labeled gold set.

Run from backend/ with the venv active:
    python3 run_eval.py | tee docs/eval_summary.txt
"""
import csv
import time

GOLD_PATH = "docs/eval_batch_labeled.csv"
DEV_IDS = {"1", "2", "3"}  # reviews already seen while developing the prompt -> not a fair test
RESULTS_PATH = "docs/eval_results.csv"
SENTIMENTS = ["positive", "neutral", "negative"]
ASPECTS = ["service", "quality", "price", "ambience"]


# ---------- loading ----------

def parse_aspects(raw):
    """'ambience, service ' -> {'ambience', 'service'}; blank -> empty set."""
    return {a.strip().lower() for a in (raw or "").split(",") if a.strip()}


def load_gold(path=GOLD_PATH):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sentiment = r["your_sentiment"].strip().lower()
            if sentiment not in SENTIMENTS:
                raise ValueError(f"Row {r['id']}: bad or missing your_sentiment: {sentiment!r}")
            rows.append({
                "id": r["id"],
                "business": r["business_name"],
                "language": r["language"].strip(),
                "text": r["review_text"],
                "gold_sentiment": sentiment,
                "gold_aspects": parse_aspects(r["your_aspects"]),
            })
    return rows


# ---------- prediction (the only part that calls Groq) ----------

def predict_all(rows, few_shot):
    from ai_utils import analyze_review  # imported here so the metrics can be tested without Groq

    preds = []
    for r in rows:
        try:
            out = analyze_review(r["text"], few_shot=few_shot)
            sentiment = str(out.get("sentiment", "")).strip().lower()
            aspects = {str(a).strip().lower() for a in out.get("aspects", [])}
            ok = sentiment in SENTIMENTS
        except Exception as e:
            print(f"  review {r['id']} failed: {e}")
            sentiment, aspects, ok = "", set(), False
        preds.append({"sentiment": sentiment, "aspects": aspects, "ok": ok})
        time.sleep(1)  # be gentle with rate limits
    return preds


# ---------- metrics ----------

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sentiment_metrics(gold, pred):
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    per_class = {}
    for c in SENTIMENTS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        support = sum(g == c for g in gold)
        per_class[c] = (*prf(tp, fp, fn), support)
    # macro-F1 only averages classes that actually appear in the gold labels
    present = [v[2] for v in per_class.values() if v[3] > 0]
    macro_f1 = sum(present) / len(present)
    return accuracy, macro_f1, per_class


def aspect_metrics(gold_sets, pred_sets):
    per_aspect = {}
    for a in ASPECTS:
        tp = sum(a in g and a in p for g, p in zip(gold_sets, pred_sets))
        fp = sum(a not in g and a in p for g, p in zip(gold_sets, pred_sets))
        fn = sum(a in g and a not in p for g, p in zip(gold_sets, pred_sets))
        per_aspect[a] = prf(tp, fp, fn)
    # micro: pool every (review, aspect) pair; invented aspects count as false positives
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    exact = sum(g == p for g, p in zip(gold_sets, pred_sets)) / len(gold_sets)
    return prf(tp, fp, fn), exact, per_aspect


# ---------- reporting ----------

def print_report(subset, rows, preds, idx):
    gold_s = [rows[i]["gold_sentiment"] for i in idx]
    pred_s = [preds[i]["sentiment"] for i in idx]
    gold_a = [rows[i]["gold_aspects"] for i in idx]
    pred_a = [preds[i]["aspects"] for i in idx]

    acc, macro_f1, per_class = sentiment_metrics(gold_s, pred_s)
    (mp, mr, mf), exact, per_aspect = aspect_metrics(gold_a, pred_a)

    print(f"\n--- {subset} (n={len(idx)}) ---")
    print(f"Sentiment: accuracy {acc:.2f} | macro-F1 {macro_f1:.2f}")
    for c, (p, r, f, support) in per_class.items():
        print(f"  {c:9s} P {p:.2f}  R {r:.2f}  F1 {f:.2f}  (support {support})")
    print(f"Aspects:   micro P {mp:.2f}  R {mr:.2f}  F1 {mf:.2f} | exact-match {exact:.2f}")
    print("  per-aspect F1: " + "  ".join(f"{a} {v[2]:.2f}" for a, v in per_aspect.items()))

    print("  confusion (rows = your label, cols = model):")
    print("            " + "  ".join(f"{c[:4]:>4s}" for c in SENTIMENTS) + "  other")
    for g in SENTIMENTS:
        counts = [sum(a == g and b == c for a, b in zip(gold_s, pred_s)) for c in SENTIMENTS]
        other = sum(a == g and b not in SENTIMENTS for a, b in zip(gold_s, pred_s))
        print(f"  {g:9s} " + "  ".join(f"{n:4d}" for n in counts) + f"  {other:5d}")


def save_results(rows, results, path=RESULTS_PATH):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "business", "language", "text", "gold_sentiment", "gold_aspects",
                    "zs_sentiment", "zs_aspects", "fs_sentiment", "fs_aspects",
                    "zs_correct", "fs_correct"])
        zs, fs = results["zero-shot"], results["few-shot"]
        for i, r in enumerate(rows):
            w.writerow([
                r["id"], r["business"], r["language"], r["text"],
                r["gold_sentiment"], ",".join(sorted(r["gold_aspects"])),
                zs[i]["sentiment"], ",".join(sorted(zs[i]["aspects"])),
                fs[i]["sentiment"], ",".join(sorted(fs[i]["aspects"])),
                zs[i]["sentiment"] == r["gold_sentiment"],
                fs[i]["sentiment"] == r["gold_sentiment"],
            ])
    print(f"\nPer-review results saved to {path}")


def main():
    rows = load_gold()
    print(f"Loaded {len(rows)} gold-labeled reviews")

    results = {}
    for mode, few_shot in [("zero-shot", False), ("few-shot", True)]:
        print(f"\nRunning {mode}...")
        results[mode] = predict_all(rows, few_shot)

    for mode, preds in results.items():
        failures = sum(not p["ok"] for p in preds)
        print(f"\n===== {mode.upper()} (failed/invalid outputs: {failures}) =====")
        subsets = {
            "all": lambda r: True,
            "held-out only (excludes dev rows 1-3)": lambda r: r["id"] not in DEV_IDS,
            "code-mixed": lambda r: r["language"] == "code-mixed",
            "english": lambda r: r["language"] == "english",
        }
        for name, keep in subsets.items():
            idx = [i for i, r in enumerate(rows) if keep(r)]
            print_report(name, rows, preds, idx)

    save_results(rows, results)
    print("\nNote: with n=24, one review is ~4 points of accuracy. Treat differences of 1-2 reviews as noise.")


if __name__ == "__main__":
    main()
