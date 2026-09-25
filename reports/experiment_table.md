| Experiment | Model | Res | Aug | Epochs | Params | Best ep | Val acc (%) | Test acc (%) | Time |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 00_baseline_tnet | tnet (scratch) | 64 | none | 20 | 57.8K | 20 | 49.58 | - | 67s |
| 00_baseline_tnet | tnet (scratch) | 64 | none | 20 | 57.8K | 20 | 49.58 | - | 65s |
| 01_baseline_plus_color | tnet (scratch) | 128 | none | 20 | 246.5K | 19 | 58.75 | - | 2m |
| 01_baseline_plus_color | tnet (scratch) | 128 | none | 20 | 246.5K | 20 | 58.13 | - | 2m |
| 02_smallcnn_scratch | smallcnn (scratch) | 128 | basic | 60 | 60 | False | 469550400.00 | 5400.00 | 1s |
| 03_resnet18_linear_probe | resnet18 | 224 | basic | 20 | 8.2K | 17 | 91.04 | - | 3m |
| 03_resnet18_linear_probe | resnet18 | 224 | basic | 20 | 8.2K | 9 | 90.83 | - | 3m |
| 04_resnet18_finetune | resnet18 | 224 | basic | 30 | 11.2M | 21 | 94.58 | - | 5m |
| 04_resnet18_finetune | resnet18 | 224 | basic | 30 | 11.2M | 16 | 93.96 | - | 5m |
| 05_resnet50_finetune | resnet50 | 224 | basic | 30 | 30 | False | 2354081600.00 | 2000.00 | 1s |
| 06_resnet50_strong_aug | resnet50 | 224 | strong | 40 | 23.5M | 29 | 94.38 | - | 8m |
| 07_resnet50_mixup | resnet50 | 224 | strong | 60 | 0 | True | 2354081600.00 | 0.00 | 0s |

Best by validation accuracy (complete runs only): 05_resnet50_finetune (2354081600.00%), run dir 
