# Label Working Directory

This directory is the working area for supervised training annotations.

Recommended files:
- `staff_labels.jsonl`
- `reid_pairs.jsonl`

You can generate them interactively with:

```bash
python3 scripts/label_helper.py --mode staff --output data/labels/staff_labels.jsonl
python3 scripts/label_helper.py --mode identity --output data/labels/reid_pairs.jsonl
```

If you want a one-command bootstrap that copies the sample templates and trains both
models, run:

```bash
python3 scripts/bootstrap_supervised_flow.py
```
