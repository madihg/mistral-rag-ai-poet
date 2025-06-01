from flask import Flask, render_template, request, jsonify
from mistralai import Mistral
import numpy as np
import os
from dotenv import load_dotenv
import traceback
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Mistral client
api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    logger.error("MISTRAL_API_KEY environment variable is not set")
    raise ValueError("MISTRAL_API_KEY environment variable is not set")

client = Mistral(api_key=api_key)

# Load and process the text
try:
    with open('essay.txt', 'r') as f:
        text = f.read()
    logger.info("Successfully loaded essay.txt")
except FileNotFoundError:
    logger.error("essay.txt file not found")
    raise FileNotFoundError("essay.txt file not found. Please ensure it exists in the project directory.")

# Create chunks
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
logger.info(f"Created {len(chunks)} chunks from the text")

def get_text_embedding(input):
    try:
        embeddings_batch_response = client.embeddings.create(
            model="mistral-embed",
            inputs=input
        )
        return embeddings_batch_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Create embeddings
try:
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    logger.info("Successfully created embeddings")
except Exception as e:
    logger.error(f"Error creating embeddings: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

def find_nearest_neighbors(query_embedding, embeddings, k=2):
    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    # Get indices of k nearest neighbors
    nearest_indices = np.argsort(similarities)[-k:][::-1]
    return nearest_indices

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
            
        question = data['question']
        logger.info(f"Received question: {question}")
        
        # Get question embedding
        question_embedding = np.array(get_text_embedding(question))
        
        # Search for similar chunks
        nearest_indices = find_nearest_neighbors(question_embedding, text_embeddings)
        retrieved_chunks = [chunks[i] for i in nearest_indices]
        
        # Create prompt
        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunks}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Answer:
        """
        
        # Generate response
        response = client.chat.complete(
            model="mistral-tiny",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return jsonify({
            'answer': response.choices[0].message.content,
            'context': retrieved_chunks
        })
    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logger.error(error_message)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_message}), 500

# For local development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 