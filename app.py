from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertModel
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import send_from_directory
from flask_cors import CORS
import os


# Create the Flask application
app = Flask(__name__, static_folder="build")
CORS(app)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


"""
FIRST MODEL FOR COSINE SIMILARITY
"""

# Load the BERT model and tokenizer
model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set the maximum sequence length
max_seq_length = 128


# Preprocess input and perform inference
def preprocess_and_predict(texts):
    # Tokenize and pad input texts
    input_ids = []
    for text in texts:
        encoded_text = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            pad_to_max_length=True,
        )
        input_ids.append(encoded_text)
    input_ids = tf.convert_to_tensor(input_ids)

    # Perform inference
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state

    # Reshape embeddings to 2-dimensional array
    embeddings = tf.reshape(embeddings, [embeddings.shape[0], -1])

    # Return the embeddings as the model's output
    return embeddings.numpy()


# Calculate cosine similarity
def calculate_cosine_similarity(embeddings):
    similarity = cosine_similarity(embeddings)
    return similarity


# Define the API endpoint
@app.route("/bert-cosine-similarity", methods=["POST"])
def bert_cosine_similarity():
    # Get the input texts from the request payload
    data = request.get_json()
    texts = data["texts"]

    # Preprocess the input texts and get embeddings
    embeddings = preprocess_and_predict(texts)

    # Calculate cosine similarity
    similarity_matrix = calculate_cosine_similarity(embeddings)

    # Convert the similarity matrix to a list of lists
    similarity_matrix = similarity_matrix.tolist()

    # Return the response as JSON
    response = {"similarity_matrix": similarity_matrix}
    return jsonify(response)


"""
SECOND MODEL FOR TEXT GENERATION
"""

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer2 = GPT2Tokenizer.from_pretrained(model_name)
model2 = TFGPT2LMHeadModel.from_pretrained(model_name)
# Set pad_token_id to eos_token_id
model.config.pad_token_id = model.config.eos_token_id


@app.route("/generate-text", methods=["POST"])
def generate_text():
    data = request.get_json()
    texts = data["texts"]

    # Combine the input texts
    combined_text = " ".join(texts)

    # Tokenize the combined text
    input_ids = tokenizer2.encode(combined_text, return_tensors="tf")

    # Create attention mask
    attention_mask = tf.ones_like(input_ids)

    # Generate text using the GPT-2 model
    output = model2.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )

    # Decode the generated text
    generated_text = tokenizer2.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})


# Run the Flask application
if __name__ == "__main__":
    app.run()
