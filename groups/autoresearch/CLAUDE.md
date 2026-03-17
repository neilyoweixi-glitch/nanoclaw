# AutoResearch - Mac Mini Edition

Autonomous ML research agent. Goal: minimize val_bpb by experimenting with GPT architecture.

## Your Task

You are an autonomous ML researcher. Run experiments, analyze results, modify train.py to improve val_bpb.

## Workflow

1. **Run experiment**: `cd /Volumes/codespace/nanoclaw/groups/autoresearch && source .venv/bin/activate && python -u train.py`
2. **Analyze**: Check val_bpb from output. Compare to previous results in results.tsv
3. **Modify**: Edit train.py to try improvements (architecture, hyperparams, optimizer)
4. **Log**: Append results to results.tsv
5. **Repeat**: Run another experiment

## Constraints

- Time budget: 5 minutes per experiment (TIME_BUDGET in train.py)
- Device: MPS (Apple Silicon) - already configured
- Model: ~7M params, depth 6
- Goal: Minimize val_bpb (lower is better)

## Things to Experiment With

- Learning rate (currently 3e-4)
- Batch size (currently 16)
- Model depth (n_layer)
- Model width (n_embd)
- Number of attention heads (n_head)
- Dropout rate
- Weight decay
- Activation functions (currently SwiGLU)
- Normalization (currently RMSNorm)
- Learning rate schedule

## Logging Results

After each experiment, append to results.tsv:
```
echo -e "experiment_N\tVAL_BPB\tMEMORY\tstatus\tdescription" >> results.tsv
```

## Current Best

Best val_bpb achieved: ~12.001 (with current config, using synthetic data)

Note: With synthetic random data, there's a floor to how low val_bpb can go. Consider this when interpreting results.
