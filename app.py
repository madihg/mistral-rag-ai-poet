from flask import Flask, render_template, request, jsonify
from mistralai import Mistral
import numpy as np
import faiss
import os
from dotenv import load_dotenv
import traceback
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Mistral client
api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable is not set")

client = Mistral(api_key=api_key)

# Load and process the text
try:
    with open('essay.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    raise FileNotFoundError("essay.txt file not found. Please ensure it exists in the project directory.")

# Create chunks
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_embedding(input):
    try:
        embeddings_batch_response = client.embeddings.create(
            model="mistral-embed",
            inputs=input
        )
        return embeddings_batch_response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

# Create embeddings and FAISS index
try:
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)
except Exception as e:
    print(f"Error creating embeddings or FAISS index: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    raise

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
        
        # Get question embedding
        question_embedding = np.array([get_text_embedding(question)])
        
        # Search for similar chunks
        k = 2
        D, I = index.search(question_embedding, k)
        retrieved_chunks = [chunks[i] for i in I[0]]
        
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
        print(error_message)
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 