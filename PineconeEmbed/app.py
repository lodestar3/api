from flask import Flask, request, jsonify
from embeddings.py import generate_embeddings, save_embeddings, search_embeddings

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    text = file.read().decode('utf-8')
    embeddings = generate_embeddings(text)
    save_embeddings(file.filename, embeddings)
    
    return jsonify({'message': 'File processed and embeddings saved'}), 200

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query_text = data.get('query')
    
    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400
    
    query_embedding = generate_embeddings(query_text)
    results = search_embeddings(query_embedding)
    
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)
