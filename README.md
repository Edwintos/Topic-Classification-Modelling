# XLM-RoBERTa Topic Classification
This repository contains a fine-tuned XLM-RoBERTa model for thematic classification of phrases in Dholuo and Swahili. The model is trained to categorize textual data into predefined themes based on semantic meaning.

# Dataset
The dataset consists of parallel phrases in Dholuo and Swahili, each labeled with a corresponding thematic area.
📂 Dataset Link: Download Here (Replace with actual link)

#Topic Areas
The dataset is structured with phrases belonging to different thematic areas, which include but are not limited to:
Social Setting
Agriculture and Food
Healthcare
Religion and Culture
Ceremony
Business and Finance
Automotive and Transport
Sports and Entertainment
Nature and Environment
Education and Technology
News and Media
History and Government
Each phrase is labeled with a theme, enabling the model to learn and predict categories for unseen text.

# Installation
Clone the repository and install dependencies:
 git clone https://github.com/yourusername/your-repo.git
 cd your-repo
 pip install -r requirements.txt

# Training the Model
Run the following script to train the model:
python train.py
This script loads the dataset, fine-tunes the XLM-RoBERTa model, and evaluates performance.

#Training Configuration:
Model: XLM-RoBERTa (Transformer-based multilingual model)
Epochs: 8
Batch Size: 16
Learning Rate: 3e-5 (with scheduler)
Gradient Accumulation: 2

#Evaluation Metrics
During training, the model is evaluated using:
Accuracy
Precision
Recall
F1-score
Results are stored in the results/ directory after training.

#Usage
To make predictions on new text:

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch

model_path = "./best_model"
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

def predict(phrase):
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax().item()

print(predict("Your test phrase here"))

#Contributing
Feel free to contribute by submitting pull requests or opening issues.


