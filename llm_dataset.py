# -*- coding: utf-8 -*-
"""LLM_dataset.ipynb
"""

## !pip install datasets

from datasets import load_dataset

ds = load_dataset("Anthropic/persuasion")

print(ds)

import json

import random
import json

system_message = (
    "You are a classification model trained to evaluate the persuasiveness of claims and arguments on a scale from 1 to 7. Your task is as follows:\n"
    "1. Evaluate the persuasiveness of a standalone claim and provide a score from 1 (not persuasive) to 7 (highly persuasive).\n"
    "2. After being provided with an argument supporting the claim, re-evaluate the claim's persuasiveness and provide an updated score from 1 to 7."
)

# Split the dataset into rows with non-zero and zero 'persuasiveness_metric'
non_zero_data = [row for row in ds["train"] if row['persuasiveness_metric'] != 0]
zero_data = [row for row in ds["train"] if row['persuasiveness_metric'] == 0]

# Randomly select 1100 entries from non_zero_data and 400 from zero_data
selected_non_zero = random.sample(non_zero_data, 1100) if len(non_zero_data) >= 1100 else non_zero_data
selected_zero = random.sample(zero_data, 400) if len(zero_data) >= 400 else zero_data

# Combine and shuffle the dataset
selected_data = selected_non_zero + selected_zero
random.shuffle(selected_data)

# Prepare the fine-tuning data
finetuning_data = []

for row in selected_data:
    conversation = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Evaluate the persuasiveness of the following standalone claim on a scale from 1 (not persuasive) to 7 (highly persuasive):\nClaim: {row['claim']}"
            },
            {"role": "assistant", "content": str(row['rating_initial'])},
            {
                "role": "user",
                "content": f"Now consider the following argument in support of the claim:\nArgument: {row['argument']}\n"
                "Evaluate the persuasiveness of the claim, taking the argument into account, on a scale from 1 (not persuasive) to 7 (highly persuasive)."
            },
            {"role": "assistant", "content": str(row['rating_final'])},
        ]
    }
    finetuning_data.append(conversation)

# Save the dataset to a JSONL file
output_file = "finetuning_data.jsonl"
with open(output_file, "w") as f:
    for entry in finetuning_data:
        json.dump(entry, f)
        f.write("\n")

print(f"Dataset for fine-tuning saved to {output_file}")

import random
import json


# Exclude rows already used in the fine-tuning dataset
remaining_non_zero = [row for row in non_zero_data if row not in selected_non_zero]
remaining_zero = [row for row in zero_data if row not in selected_zero]

# Randomly select 100 entries from the remaining zero_data
validation_zero = random.sample(remaining_zero, 100) if len(remaining_zero) >= 100 else remaining_zero

# Use all remaining rows from non_zero_data for validation
validation_non_zero = remaining_non_zero

# Combine and shuffle validation data
validation_data = validation_non_zero + validation_zero
random.shuffle(validation_data)

# Prepare the validation dataset
validation_finetuning_data = []

for row in validation_data:
    conversation = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Evaluate the persuasiveness of the following standalone claim on a scale from 1 (not persuasive) to 7 (highly persuasive):\nClaim: {row['claim']}"
            },
            {"role": "assistant", "content": str(row['rating_initial'])},
            {
                "role": "user",
                "content": f"Now consider the following argument in support of the claim:\nArgument: {row['argument']}\n"
                "Evaluate the persuasiveness of the claim, taking the argument into account, on a scale from 1 (not persuasive) to 7 (highly persuasive)."
            },
            {"role": "assistant", "content": str(row['rating_final'])},
        ]
    }
    validation_finetuning_data.append(conversation)

# Save the validation dataset to a JSONL file
validation_output_file = "validation_data.jsonl"
with open(validation_output_file, "w") as f:
    for entry in validation_finetuning_data:
        json.dump(entry, f)
        f.write("\n")

print(f"Validation dataset saved to {validation_output_file}")

import random
import json

# Define the system message to guide the model
system_message = (
    "You are a language model designed to generate persuasive arguments in support of given claims. "
    "Your task is to craft arguments that are coherent, compelling, and logically support the claim provided by the user."
)

# Filter rows with persuasiveness_metric > 0
filtered_data = [row for row in ds["train"] if row['persuasiveness_metric'] > 0]

# Shuffle the filtered data to ensure randomness
random.shuffle(filtered_data)

# Perform a 90-10 train-validation split
split_index = 1100
train_data = filtered_data[:split_index]
validation_data = filtered_data[split_index:]

# Prepare fine-tuning data for training
finetuning_data_train = []
for row in train_data:
    conversation = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Claim: {row['claim']}"
            },
            {
                "role": "assistant",
                "content": row['argument']  # Assuming 'argument' is the ground truth argument
            },
        ]
    }
    finetuning_data_train.append(conversation)

# Save training data to a JSONL file
output_file_train = "finetuning_data_generation.jsonl"
with open(output_file_train, "w") as f:
    for entry in finetuning_data_train:
        json.dump(entry, f)
        f.write("\n")

print(f"Training dataset saved to {output_file_train}")

# Prepare fine-tuning data for validation
finetuning_data_validation = []
for row in validation_data:
    conversation = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Claim: {row['claim']}"
            },
            {
                "role": "assistant",
                "content": row['argument']  # Assuming 'argument' is the ground truth argument
            },
        ]
    }
    finetuning_data_validation.append(conversation)

# Save validation data to a JSONL file
output_file_validation = "validation_data_generation.jsonl"
with open(output_file_validation, "w") as f:
    for entry in finetuning_data_validation:
        json.dump(entry, f)
        f.write("\n")

print(f"Validation dataset saved to {output_file_validation}")