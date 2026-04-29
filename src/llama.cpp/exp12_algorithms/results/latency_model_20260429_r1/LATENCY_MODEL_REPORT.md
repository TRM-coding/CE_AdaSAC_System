# Latency Model Report

This report fits sklearn tabular regressors to the measured exp12 latency data.
The current measured data covers CPU-set and load-vector variation for one fixed workload; `Q_pct` is present in the feature interface but fixed in this run.

## Dataset

- rows: 164
- train rows: 123
- test rows: 41
- CPUs: 60,61,62,63,64,65,66,67

## Test Metrics

| model | MAE ms | MAPE | median APE | p90 APE | max APE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| extra_trees_log | 31.4740 | 0.1180 | 0.0299 | 0.1825 | 2.0736 | 0.4095 |
| dummy_median | 60.1338 | 0.5068 | 0.0467 | 0.8982 | 3.2603 | -0.0001 |
| ridge_log | 85.7062 | 0.7434 | 0.1156 | 0.4341 | 23.0379 | -2.0158 |

## Best Model

- model: `extra_trees_log`
- test MAE: 31.4740 ms
- test MAPE: 0.1180
- test median APE: 0.0299
- test p90 APE: 0.1825

## Worst Test Predictions

| source | name | actual ms | predicted ms | APE |
|---|---|---:|---:|---:|
| heterogeneous | 2core_0_100 | 86.2750 | 265.1772 | 2.0736 |
| heterogeneous | 4core_0_20_60_100 | 1319.1255 | 410.9698 | 0.6885 |
| uniform_multi | multi_60_61_62_63_64_65_66_67_load40 | 38.8865 | 51.7003 | 0.3295 |
| uniform_multi | multi_60_61_62_63_64_65_load30 | 36.5985 | 46.3495 | 0.2664 |
| single | single_cpu65_load100 | 161.3040 | 190.7365 | 0.1825 |
| uniform_multi | multi_60_61_62_63_64_65_load40 | 37.7437 | 42.7908 | 0.1337 |
| single | single_cpu65_load90 | 157.6940 | 178.0915 | 0.1293 |
| uniform_multi | multi_61_63_load100 | 87.6020 | 95.5849 | 0.0911 |
| uniform_multi | multi_61_63_load40 | 82.1405 | 89.2261 | 0.0863 |
| single | single_cpu61_load0 | 154.4350 | 141.6607 | 0.0827 |

## Files

- `supervised_dataset.csv`: normalized measured dataset and train/test split
- `model_metrics.csv`: train/test metrics for all candidate models
- `test_predictions.csv`: held-out predictions from the best model
- `best_model.pkl`: pickled sklearn model plus feature metadata
