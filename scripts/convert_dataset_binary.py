import pandas as pd
import json

# Load the train and test TSV files
train_file_path = "../data/claudette_train_merged_without_pinc.tsv"
test_file_path = "../data/claudette_test_merged_without_pinc.tsv"

# Load training data
df_train = pd.read_csv(train_file_path, sep="\t")

# Define the task definition
task_definition = (
    "You are an expert in identifying potentially unfair clauses in terms of service documents. "
    "Answer 'Yes' or 'No' only, followed by a brief reason if 'Yes'. "
    "Is the following sentence a potentially unfair clause?"
)

# Separate Positive and Negative Examples
positive_examples = []
negative_examples = []

for _, row in df_train.iterrows():
    example = {
        "input": row["text"],  # Assuming "text" contains the input
        "output": "Yes" if row["label"] == 1 else "No"  # Map labels to 'Yes' or 'No'
    }
    if row["label"] == 1:
        positive_examples.append(example)
    else:
        negative_examples.append(example)

# Load test data
df_test = pd.read_csv(test_file_path, sep="\t")

# Add test data to Instances
instances = []
for _, row in df_test.iterrows():
    instance = {
        "input": row["text"],
        "output": ["Yes" if row["label"] == 1 else "No"]
    }
    instances.append(instance)

# Create the JSON structure
task_data = {
    "Definition": task_definition,
    "Positive Examples": positive_examples,
    "Negative Examples": negative_examples,
    "Instances": instances
}

# Save the JSON file
output_path = "converted_data.json"
with open(output_path, "w") as json_file:
    json.dump(task_data, json_file, indent=4)

print(f"Data successfully converted and saved to {output_path}")
