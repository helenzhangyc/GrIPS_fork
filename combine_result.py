import numpy as np
import pandas as pd

# Data from your table (replace with actual values from your dataset)
# No viable candidates found for 'J'
# No viable candidates found for 'LTD'
data = {
    "Label": ["UNFAIR", "A", "CH", "CR", "LAW", "TER", "USE"],
    "Accuracy": [0.6, 0.68, 0.56, 0.56, 0.64, 0.6, 0.32],
    "F1": [0.375, 0.40, 0.36, 0.36, 0.39, 0.375, 0.24],
    "Precision": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "Recall": [0.3, 0.34, 0.28, 0.28, 0.32, 0.3, 0.16],
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Calculate overall accuracy (mean of the accuracies in this case)
overall_accuracy = np.mean(df["Accuracy"])

# Calculate macro-averages
macro_f1 = np.mean(df["F1"])
macro_precision = np.mean(df["Precision"])
macro_recall = np.mean(df["Recall"])

# Create a summary dictionary
summary = {
    "Multilabel Accuracy": overall_accuracy,
    "Macro F1": macro_f1,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
}

# Convert to DataFrame for better display
summary_df = pd.DataFrame([summary])

# Display the summary
print("Multilabel Metrics Summary:")
print(summary_df)
