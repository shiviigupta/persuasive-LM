"""## Inference on finetuned models"""

import json
import openai
from sklearn.metrics import accuracy_score, cohen_kappa_score


openai.api_key = "" ## REPLACE
fine_tuned_model = "" ##REPLACE

validation_file = "validation_data.jsonl"
validation_data = []
with open(validation_file, "r") as f:
    for line in f:
        validation_data.append(json.loads(line))

def map_prediction(response):
    lastchars = response[-5:]
    lastchars2 = response[-30:].lower()
    if '1' in lastchars or 'Strongly oppose'.lower() in lastchars2:
        return 1
    elif '2' in lastchars or 'Oppose'.lower() in lastchars2:
        return 2
    elif '3' in lastchars or 'Somewhat oppose'.lower() in lastchars2:
        return 3
    elif '4' in lastchars or 'Neither oppose nor support'.lower() in lastchars2:
        return 4
    elif '5' in lastchars or 'Somewhat support'.lower() in lastchars2:
        return 5
    elif '6' in lastchars or 'support'.lower() in lastchars2:
        return 6
    elif '7' in lastchars or 'Strongly support'.lower() in lastchars2:
        return 7
    else:
        print("malformed")
        print(response)
        return -1

# Initialize ground truth and predictions
true_initial = []
true_final = []
pred_initial = []
pred_final = []

# Generate predictions using the fine-tuned model
for entry in validation_data:
    messages = entry["messages"]

    # Extract system prompt
    system_message = messages[0]["content"]

    # Extract claim for the first prediction
    claim_prompt = messages[1]["content"]

    # Extract argument for the second prediction
    argument_prompt = messages[3]["content"]

    # Ground truth: Map assistant content (initial and final ratings)
    true_initial.append(map_prediction(messages[2]["content"]))
    true_final.append(map_prediction(messages[4]["content"]))

    # Generate predictions for the initial rating
    try:
        response_initial = openai.ChatCompletion.create(
            model=fine_tuned_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": claim_prompt},
            ],
        )
        pred_initial_response = response_initial["choices"][0]["message"]["content"]
        pred_initial.append(map_prediction(pred_initial_response))
    except Exception as e:
        print(f"Error generating initial response: {e}")
        pred_initial.append(-1)

    # Generate predictions for the final rating
    try:
        response_final = openai.ChatCompletion.create(
            model=fine_tuned_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": claim_prompt},
                {"role": "assistant", "content": pred_initial_response},
                {"role": "user", "content": argument_prompt},
            ],
        )
        pred_final_response = response_final["choices"][0]["message"]["content"]
        pred_final.append(map_prediction(pred_final_response))
    except Exception as e:
        print(f"Error generating final response: {e}")
        pred_final.append(-1)

# Filter out malformed predictions
valid_indices_initial = [i for i, pred in enumerate(pred_initial) if pred != -1]
valid_indices_final = [i for i, pred in enumerate(pred_final) if pred != -1]

true_initial = [true_initial[i] for i in valid_indices_initial]
pred_initial = [pred_initial[i] for i in valid_indices_initial]

true_final = [true_final[i] for i in valid_indices_final]
pred_final = [pred_final[i] for i in valid_indices_final]

# Calculate accuracy and Cohen's kappa for both
accuracy_initial = accuracy_score(true_initial, pred_initial)
accuracy_final = accuracy_score(true_final, pred_final)

kappa_initial = cohen_kappa_score(true_initial, pred_initial)
kappa_final = cohen_kappa_score(true_final, pred_final)

p_score = np.array(pred_final) - np.array(pred_initial)
true_p_score =  np.array(true_final) - np.array(true_initial)

accuracy_p_score = accuracy_score(true_p_score, p_score)
kappa_p = cohen_kappa_score(true_p_score, p_score)

print(f"Accuracy for rating_initial: {accuracy_initial:.2f}")
print(f"Accuracy for rating_final: {accuracy_final:.2f}")
print(f"Cohen's kappa for rating_initial: {kappa_initial:.2f}")
print(f"Cohen's kappa for rating_final: {kappa_final:.2f}")
print(f"Accuracy for p score: {accuracy_p_score:.2f}")
print(f"Cohen's kappa for p score: {kappa_p:.2f}")