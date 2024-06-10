import openai
import pinecone

# OpenAIとPineconeのAPIキーを設定
openai.api_key = 'YOUR_OPENAI_API_KEY'

pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='YOUR_PINECONE_ENVIRONMENT')
index_name = 'your-index'
pinecone_index = pinecone.Index(index_name)

def generate_embeddings(text):
    response = openai.Embedding.create(
        model='text-embedding-ada-002',  # 使用するモデルを指定
        input=text
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

def save_embeddings(embedding_id, embeddings):
    pinecone_index.upsert(
        vectors=[(embedding_id, embeddings)]
    )

def search_embeddings(query_embedding, top_k=10):
    results = pinecone_index.query(
        queries=[query_embedding],
        top_k=top_k
    )
    return results
