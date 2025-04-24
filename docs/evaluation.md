[Back to Main README](../README.md)

---

## Model Evaluation Metrics

This document outlines the key evaluation metrics used to assess the performance of the AI-based Patient Appointment prediction model.

### Metrics Tracked

| Metric                | Description                                                                                               |
|-----------------------|-----------------------------------------------------------------------------------------------------------|
| **Subset Accuracy**   | Measures the exact match ratio between predicted and true labels.                                         |
| **Hamming Loss**      | Fraction of labels that are incorrectly predicted (either wrongly included or excluded). Lower is better. |
| **Precision**         | Proportion of true positive predictions out of all positive predictions made by the model.                |
| **Recall**            | Proportion of true positive predictions out of all actual positives in the data.                          |
| **F1-Score**          | Harmonic mean of Precision and Recall. Balances both metrics.                                             |

### Graphical Analysis

Model metrics over training runs:

![Trends Across Multiple Runs](/output_images/train/training_plot.png)

You can visualize the trends of these metrics across multiple runs:

- **Subset Accuracy** and **F1-Score** showed a gradual increase, stabilizing around 94% and 96% respectively.
- **Hamming Loss** remained low (below 1%), indicating minimal label prediction errors.
- **Precision** and **Recall** maintained a healthy balance, supporting the model’s robustness.

### Interpretation

The model demonstrates:

- **High Accuracy**: Achieving over 94% subset accuracy indicates reliable predictions for doctor appointments.

- **Low Hamming Loss**: The model rarely predicts incorrect labels, with an error rate under 1%.

- **Balanced Precision and Recall**: Ensures the model not only predicts appointments correctly but also minimizes missed or false predictions.

- **Strong F1-Score**: A score close to 0.97 confirms a balanced trade-off between precision and recall.

These metrics suggest the model is **sufficiently reliable** for production-level use. However, ongoing monitoring and periodic retraining with new data are recommended to maintain performance.

### Dual-Axis Plot for Hamming Loss

![Plot for Hamming Loss](/output_images/train/dual_axis_hamming_plot.png)

#### Detailed Interpretation:

1. Subset Accuracy (Blue):
- Stable early on (~94-95%) but:

- Experiences fluctuations between Run_10 and Run_24 (drops to ~90-92%).

- Run_13 and Run_24 show noticeable dips (~92% and ~90% respectively).

```bash
Insight:
Subset accuracy is strict (requires perfect label matches), so small data/model variations can impact it more.
```

2. Precision (Orange):
- High and stable (~98-99%) for most runs.

- A sharp drop around Run_13 (~94%), and another dip at Run_24 (~90%).

```bash
Insight:
Precision is generally robust, but specific runs (e.g., Run_13, Run_24) face challenges likely due to data splits or minor model instability.
```

3. Recall (Green):
- Fairly consistent (~96-97%), with moderate variance.

- Peaks at Run_12 and Run_16 (~98%), but shows variability in the later runs.

```bash
Insight:
Recall remains more stable than precision in fluctuating runs, suggesting that the model tends to predict all required labels, even if some are incorrect.
```

4. F1 Score (Red):
- Follows precision/recall trend closely (~97-98%).

- Slight dips in runs with low precision or recall, confirming balanced performance.

5. Hamming Loss (Purple, Right Axis):
- Generally low (~0.006 - 0.009), indicating minimal label-level errors.

- Inverse correlation with Subset Accuracy:

- When Subset Accuracy drops (e.g., Run_13, Run_24), Hamming Loss rises.

```bash
Insight:
Even if the exact set of labels (subset accuracy) isn't perfect, individual label predictions remain mostly correct.
```

---
[Back to Main README](../README.md)