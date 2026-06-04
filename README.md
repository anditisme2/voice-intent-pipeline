# Voice Intent Pipeline

A deep learning pipeline for structured slot extraction from natural language food and restaurant ordering commands.

Built from scratch as a learning project — no HuggingFace, no pretrained weights. Every component is hand-rolled in PyTorch. 

## What it does

Takes a free-form utterance:
```json
{ "text": "I want something warm and spicy, I'm feeling very tired" }
```
And returns structured slot predictions:
```json
{ "craving": "spicy", "mood": "tired", "temp_preference": "warm" }
```

## Stack

- **Model**: Transformer encoder with token-level classification head (BIO tagging)
- **Data**: Synthetically generated via Claude API, ~10,000 labelled utterances
- **Training**: PyTorch with MPS acceleration (Apple Silicon M-series)
- **Decoding**: BIO span decoder → JSON slot dict

## Pipeline stages

| Stage | Description |
|---|---|
| 1. Data generation | LLM-generated (utterance, slot dict) pairs |
| 2. Tokenization + BIO tagging | Custom tokenizer, slot span labelling |
| 3. Dataset / DataLoader | Padding, batching, train/val split |
| 4. Transformer encoder | Embeddings, multi-head attention, FFN, positional encoding |
| 5. Classification head | Token-level linear layer over BIO label set |
| 6. Training loop | AdamW, scheduler, slot F1 evaluation, early stopping |
| 7. Inference | BIO decode → structured JSON output |

## Slots

`craving` · `temp_preference` · `mood` · `cuisine` · `portion_size` · `dietary`
