# Data dictionary

## `A1_SMS_test_1000.csv`
- `sms_id`: stable test-sample ID, e.g. `SMS0001`
- `text`: raw SMS text supplied to both prompting methods
- `label`: ground truth (`spam` or `ham`)
- `source_row`: one-based row position in the downloaded UCI SMS text file

## `A1_3shot_examples_FINAL.csv`
- `example_id`: `EX01`–`EX03`
- `text`: actual fixed prompt demonstration SMS
- `label`: ground truth (`spam` or `ham`)
- `source_row`: source row in the downloaded text file

## `A1_official_results.csv`
- `sms_id`: foreign key to `A1_SMS_test_1000.csv`
- `true_label`: original ground-truth class (`spam` or `ham`)
- `method`: `Zero-shot` or `3-shot`
- `prediction`: original *strict-parser* result (`Spam`, `Not Spam`, `INVALID`)
- `raw_output`: exactly recorded model output string
- `input_tokens`: prompt tokens measured by tokenizer
- `output_tokens`: generated tokens
- `time_seconds`: elapsed wall-clock time around generation (includes GPU sync)

No confidence/probability scores were recorded. Token counts and per-call times are **measured artifacts**, not guarantees of API pricing or production latency.
