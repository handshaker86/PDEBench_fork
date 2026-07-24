# Rebuttal metrics

Evaluate periodic PDEBench velocity fields exported as `frame_i -> (u, v)`:

```powershell
python -m pdebench.models.rebuttal_evaluate --target target.npz --prediction prediction/FNO/dataset/prediction_result.npz --output-dir results/rebuttal --dx 0.0078125 --dy 0.0078125
```

The evaluator uses periodic central differences and reports absolute ground
truth/predicted divergence as well as divergence-field mismatch. The module
also exposes `continuity_residual` for compressible density/velocity arrays.
