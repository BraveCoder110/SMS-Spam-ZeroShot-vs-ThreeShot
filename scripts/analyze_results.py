import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(filename):
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalized(raw):
    value = raw.strip()
    if value.startswith("Label:"):
        value = value[len("Label:"):].strip()
    return value if value in ("Spam", "Not Spam") else "INVALID"


def metrics(rows):
    tp = fp = tn = fn = 0
    for r in rows:
        truth = "Spam" if r["true_label"] == "spam" else "Not Spam"
        pred = normalized(r["raw_output"])
        if pred == "INVALID":
            raise ValueError(f"Unparsed output: {r['sms_id']}: {r['raw_output']!r}")
        if truth == "Spam" and pred == "Spam": tp += 1
        elif truth == "Spam" and pred == "Not Spam": fn += 1
        elif truth == "Not Spam" and pred == "Spam": fp += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "n": len(rows), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": (tp + tn) / len(rows), "precision_spam": precision,
        "recall_spam": recall, "f1_spam": f1,
        "format_noncompliant": sum(r["prediction"] == "INVALID" for r in rows),
        "input_tokens_total": sum(int(r["input_tokens"]) for r in rows),
        "input_tokens_mean": sum(int(r["input_tokens"]) for r in rows) / len(rows),
        "seconds_total": sum(float(r["time_seconds"]) for r in rows),
        "seconds_mean": sum(float(r["time_seconds"]) for r in rows) / len(rows),
    }


def correct(r):
    expected = "Spam" if r["true_label"] == "spam" else "Not Spam"
    return normalized(r["raw_output"]) == expected


def main():
    test = read_csv("A1_SMS_test_1000.csv")
    demos = read_csv("A1_3shot_examples_FINAL.csv")
    rows = read_csv("A1_official_results.csv")
    assert len(test) == 1000 and Counter(x["label"] for x in test) == {"ham": 500, "spam": 500}
    assert len(demos) == 3 and Counter(x["label"] for x in demos) == {"ham": 1, "spam": 2}
    assert len(set(x["text"] for x in test)) == len(test)
    assert not set(x["text"] for x in test) & set(x["text"] for x in demos)
    test_labels = {r["sms_id"]: r["label"] for r in test}
    assert len(rows) == 2000
    assert all(r["sms_id"] in test_labels and r["true_label"] == test_labels[r["sms_id"]] for r in rows)
    pair = {}
    for r in rows:
        key = (r["sms_id"], r["method"])
        assert key not in pair, f"Duplicate prediction: {key}"
        pair[key] = r
    assert all((sid, method) in pair for sid in test_labels for method in ["Zero-shot", "3-shot"])

    group = {name: metrics([r for r in rows if r["method"] == name]) for name in ["Zero-shot", "3-shot"]}
    paired = Counter()
    for sid in test_labels:
        z = correct(pair[(sid, "Zero-shot")])
        t = correct(pair[(sid, "3-shot")])
        paired[(z, t)] += 1
    b = paired[(False, True)]  # zero wrong, three correct
    c = paired[(True, False)]  # zero correct, three wrong
    n = b + c
    # Exact two-sided McNemar/binomial test, valid for paired binary successes.
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))
    summary = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "posthoc_output_normalization": "strip one optional leading 'Label:' prefix",
        "metrics": group,
        "paired": {
            "both_correct": paired[(True, True)],
            "zero_wrong_three_correct": b,
            "zero_correct_three_wrong": c,
            "both_wrong": paired[(False, False)],
            "mcnemar_exact_two_sided_p": p,
        },
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
