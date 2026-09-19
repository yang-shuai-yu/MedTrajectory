# Original Delphi Patient Future Demo Reference

This directory is a read-only copy from:

```text
the original Delphi research workspace
```

It is kept as a patient-level reference artifact for the original Delphi-style demo. It should not be treated as the `Exp2 multitype internal anchor` row in the multi-patient rollout benchmark.

## Boundary

- Source model/demo lineage: original Delphi-style patient future demo from the read-only `Delphi_rec` workspace.
- Demo case: `AF-FUTURE-001`.
- Use in report: reference-only qualitative demo artifact.
- Not comparable as: same-split, same-patient multi-patient rollout baseline.

For numeric official Delphi-2M demo reference, use:

```text
results/delphi2m_original_demo/STATUS.md
results/model_compare_explicit_split/delphi_demo_val.json
```

The official Delphi-2M reference numbers currently used in the report are:

| Reference | Dataset | Split | Patients | Eval targets | Top1 | Top5 | Top10 | Notebook mean AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Delphi-2M original demo | `ukb_simulated_data` | val | 7144 | 171903 | 0.0448 | 0.1449 | 0.2293 | 0.7565 |

