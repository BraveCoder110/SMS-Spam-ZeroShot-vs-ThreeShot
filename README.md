# SMS Spam Classification: A Comparison of Zero-Shot and 3-Shot Prompting

This repository documents a comparative experiment designed to evaluate the performance of zero-shot versus 3-shot prompting on a dataset of 1,000 labeled SMS messages. The experiment utilizes the **Qwen3-4B-Instruct** Large Language Model, with consistent inference parameters applied throughout. The experiment was conducted on the Google Colab platform, utilizing an NVIDIA Tesla T4 GPU.

```text
SMS_Spam_Prompting/
├── README.md
├── data/
│   ├── A1_SMS_test_1000.csv          # Frozen evaluation set, 500 spam + 500 ham
│   ├── A1_3shot_examples_FINAL.csv   # Three fixed demonstrations, 2 spam + 1 ham
│   └── A1_official_results.csv       # 2,000 saved predictions with original outputs
├── docs/
│   └── A1_sampling_audit.json        # Original sampling audit
├── scripts/
│   ├── analyze_results.py           # Standard-library-only results verification
│   └── reproduce_inference.py       # Optional GPU inference implementation
└── results_summary.json             # Calculated from the saved results CSV
```