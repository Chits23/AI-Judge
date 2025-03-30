import streamlit as st
import torch
import json

from transformers import BertTokenizer, BertForSequenceClassification

# Function to load labels from JSONL file
def load_labels(file_path):
    all_labels = []
    with open(file_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            all_labels.append(obj.get("id"))
    return all_labels

# Function to make predictions
def predict_classes(model, tokenizer, crime_description, all_labels):
    inputs = tokenizer(crime_description, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_classes = torch.sigmoid(logits) > 0.5
    predicted_labels = [all_labels[i] for i, label in enumerate(predicted_classes[0]) if label == 1]
    return predicted_labels

# Load the fine-tuned BERT model
model_path = "./fine_tuned_model"
model_name = "nlpaueb/legal-bert-small-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set Streamlit app title and description
st.title("AI Judge 🤖")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Prediction"])

if page == "Home":
    # Display home page content
    st.image("ai_judge.jpg")
    st.write("""
    AI Judge is an application designed to assist in crime classification using Natural Language Processing (NLP) techniques.

    ### How it works:
    - **Crime Description Input:** Users can input a description of a crime into the text area provided.
    - **Prediction:** After submitting the crime description, the application uses a fine-tuned BERT model to predict the relevant sections of law (e.g., IPC sections) that may be applicable to the crime.
    - **Result Display:** The predicted sections of law are displayed, providing users with insights into the legal aspects of the crime.

    ### Benefits:
    - **Efficiency:** Quickly identifies relevant sections of law based on crime descriptions.
    - **Accuracy:** Utilizes advanced NLP techniques and a fine-tuned BERT model for precise predictions.
    - **Accessibility:** Easy-to-use interface accessible to legal professionals, law enforcement personnel, and the general public.

    ### Disclaimer:
    AI Judge is intended for informational purposes only and should not be relied upon as legal advice. Always consult with a qualified legal professional for legal matters.
    """)
    

elif page == "Prediction":
    # Load labels
    all_labels = load_labels("secs.jsonl")

    # Text area for user input
    crime_description = st.text_area("Crime Description", "")

    # Submit button to trigger prediction
    if st.button("Submit"):
        # Make predictions
        predicted_labels = predict_classes(model, tokenizer, crime_description, all_labels)
        # Display predicted classes
        st.write("Sections to be applied:")
        for label in predicted_labels:
            st.write(label)
