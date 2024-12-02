import pandas as pd
import json

# Load the TSV file
file_path = "../data/claudette_train_merged.tsv"
df = pd.read_csv(file_path, sep="\t")

# Define the task definition
task_definition = "You are an expert in identifying potentially unfair clauses in terms of service documents. Answer 'Yes' or 'No' only, followed by a brief reason if 'Yes'. Is the following sentence a potentially unfair clause?"

# Separate Positive and Negative Examples
positive_examples = []
negative_examples = []

for _, row in df.iterrows():
    example = {
        "input": row["text"],  # Assuming "text" contains the input
        "output": [str(row["label"])]  # Convert label to string
    }
    if row["label"] == 1:
        positive_examples.append(example)
    elif row["label"] == 0:
        negative_examples.append(example)

# Create the JSON structure
task_data = {
    "Definition": task_definition,
    "Positive Examples": positive_examples,
    "Negative Examples": negative_examples,
    "Instances": []  # Keeping this empty as per GrIPS requirements for now
}

# Add all data rows to Instances (if needed)
for _, row in df.iterrows():
    instance = {
        "input": row["text"],
        "output": [str(row["label"])]
    }
    task_data["Instances"].append(instance)

# Save the JSON file
output_path = "converted_data.json"
with open(output_path, "w") as json_file:
    json.dump(task_data, json_file, indent=4)

print(f"Data successfully converted and saved to {output_path}")
