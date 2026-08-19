# Processed data

`go_emotions_single_label.jsonl` contains GoEmotions examples that have **exactly one** emotion label. Train, validation, and test splits from the original dataset are merged into this single file.

Each line is a JSON object:

```json
{"id": "eebbqej", "text": "My favourite food is anything I didn't have to cook myself.", "label": "neutral"}
```

| Field | Description |
|-------|-------------|
| `id` | Original GoEmotions comment id |
| `text` | Reddit comment text |
| `label` | One of 28 emotion names (27 GoEmotions labels + `neutral`) |

Regenerate with:

```bash
python scripts/pull_go_emotions.py
```
