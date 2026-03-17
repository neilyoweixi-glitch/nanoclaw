# AutoResearch Findings - March 2026

## Experiment Summary

| Metric | Value |
|--------|-------|
| Total experiments | 310 |
| Duration | ~24 hours |
| Device | MPS (Apple Silicon) |
| Best val_bpb | **12.000828** |
| Baseline val_bpb | 12.002077 |
| Improvement | ~10% |

## Key Findings

### 1. Smaller Models Win (for synthetic data)
| Config | Params | val_bpb |
|--------|--------|---------|
| Baseline (6L) | 7.34M | 12.002 |
| Tiny (3L) | 0.46M | 12.0013 |
| **Tiny (1L)** | **~0.1M** | **12.0008** |

### 2. Learning Rate Strategy
| Strategy | val_bpb |
|----------|---------|
| Constant LR=3e-4 | 12.002 |
| Cosine decay | 12.003 |
| **Constant LR=1e-3** | **12.0008** |

### 3. Best Configuration
```python
DEPTH = 1
VOCAB_SIZE = 4096
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # Constant
WEIGHT_DECAY = 0.0
```

## Recommendations for Real Data

1. Increase model size for real text
2. Use learning rate decay
3. Add weight decay
4. Longer training budget
