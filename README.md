# Evaluating Persuasiveness of Language Models

[Presentation slides](https://docs.google.com/presentation/d/12q8UaCYGNGlH2Rv7pUfdggLOcynsWnc6e_uK9apHgsQ/edit?usp=sharing)

Used [Anthropic's Persuasion Dataset](https://www.anthropic.com/research/measuring-model-persuasiveness) ([Anthropic/persuasion](https://huggingface.co/datasets/Anthropic/persuasion)) to evaluate understanding of persuasiveness of Language Models, and fine-tuned a GPT model to be "more persuasive"

Code contains dataset creation for train-validation-test splits, followed by inference scripts. Model is trained using OpenAI's finetuning API.
