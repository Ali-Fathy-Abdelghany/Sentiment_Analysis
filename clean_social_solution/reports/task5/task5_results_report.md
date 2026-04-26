
# Task 5 Results Report

## 1. Introduction
This report benchmarks the optimized sentiment model from Task 4 against a Hugging Face SOTA model on the same labeled dataset.
The comparison is performed on an identical test split and includes per-category analysis to assess category-specific strengths/weaknesses.

## 2. Methodology
- Optimized model selected from Task 4 results: s3_tfidf_svc
- SOTA model used: siberett/roberta-sentiment-analysis-finetune
- Evaluation labels: ['Negative', 'Neutral', 'Positive']
- Test split settings: random_state=42, test_size=0.2, stratified by ground truth
- SOTA inference done with transformers pipeline('text-classification', model=...)
- Label mapping logic:
  - semantic mapping for labels containing neg/neu/pos
  - numeric-label mapping for star-like outputs
  - binary fallback with confidence-to-neutral conversion (threshold=0.60) when needed
- Category tags sourced from Task 2 preprocessing output for the selected scheme

## 3. Quantitative Results
### Side-by-side summary
| model                                               |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr | confusion_matrix                      |
|:----------------------------------------------------|-----------:|------------------:|---------------:|-----------:|--------------:|:--------------------------------------|
| Optimized (s3_tfidf_svc)                            |     0.725  |          0.77752  |       0.69127  |   0.696283 |      0.824498 | [[33, 2, 0], [5, 15, 1], [11, 3, 10]] |
| SOTA (siberett/roberta-sentiment-analysis-finetune) |     0.7875 |          0.808182 |       0.771429 |   0.780154 |      0.89318  | [[31, 3, 1], [5, 16, 0], [5, 3, 16]]  |

### Per-category (Accuracy and Macro-F1)
| category          |   n_samples |   optimized_acc |   optimized_f1 |   sota_acc |   sota_f1 |
|:------------------|------------:|----------------:|---------------:|-----------:|----------:|
| delivery_shipping |          23 |        0.695652 |       0.624668 |   0.73913  |  0.711111 |
| other             |          22 |        0.772727 |       0.661472 |   0.772727 |  0.735229 |
| app_usability     |          18 |        0.611111 |       0.571958 |   0.888889 |  0.890993 |
| pricing_discount  |           7 |        0.714286 |       0.30303  |   0.571429 |  0.266667 |
| refund_return     |           5 |        0.8      |       0.444444 |   0.8      |  0.761905 |
| ads_promotions    |           2 |        1        |       1        |   1        |  1        |
| customer_service  |           1 |        1        |       1        |   1        |  1        |
| product_quality   |           1 |        1        |       1        |   1        |  1        |
| scam_trust        |           1 |        1        |       1        |   1        |  1        |

### Best category gains for SOTA (F1 difference)
| category          |   n_samples |   optimized_f1 |   sota_f1 |   f1_gain |
|:------------------|------------:|---------------:|----------:|----------:|
| app_usability     |          18 |       0.571958 |  0.890993 | 0.319036  |
| refund_return     |           5 |       0.444444 |  0.761905 | 0.31746   |
| delivery_shipping |          23 |       0.624668 |  0.711111 | 0.0864427 |
| other             |          22 |       0.661472 |  0.735229 | 0.0737567 |
| ads_promotions    |           2 |       1        |  1        | 0         |

### Largest drops for SOTA (F1 difference)
| category         |   n_samples |   optimized_f1 |   sota_f1 |    f1_gain |
|:-----------------|------------:|---------------:|----------:|-----------:|
| pricing_discount |           7 |        0.30303 |  0.266667 | -0.0363636 |
| product_quality  |           1 |        1       |  1        |  0         |
| customer_service |           1 |        1       |  1        |  0         |
| ads_promotions   |           2 |        1       |  1        |  0         |
| scam_trust       |           1 |        1       |  1        |  0         |

## 4. Visual Analysis
The following plots were generated and saved:
- overall_metrics_comparison.png
- per_category_accuracy.png
- per_category_f1.png
- confusion_matrices_side_by_side.png
- error_rate_by_category_heatmap.png

Interpretation guide:
- Overall metrics chart indicates global performance dominance.
- Per-category bars reveal category-specific robustness differences.
- Confusion matrices show which sentiment classes are most confused by each model.
- Error-rate heatmap highlights category-level failure concentration.

## 5. Conclusion
This benchmark provides a direct comparison between a task-optimized in-project model and a transferable SOTA transformer model.
Use the per-category and confusion analysis to decide whether to keep one global model, ensemble both, or route by category.
