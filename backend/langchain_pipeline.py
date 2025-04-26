import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# --- Retriever Class ---

class Retriever:
    def __init__(self, index, texts, embedding_model):
        self.index = index
        self.texts = texts
        self.embedding_model = embedding_model

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), k)
        return [self.texts[i] for i in indices[0]]

# --- Generator Class ---

class Generator:
    def __init__(self, api_key, model_name="text-davinci-003"):
        openai.api_key = api_key
        self.model_name = model_name

    def generate(self, context, query):
        prompt = f"""You are a helpful assistant that recommends places.

Context:
{context}

Question: {query}

Answer:"""
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()

# --- RAG Pipeline Class ---

class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query, k=3):
        contexts = self.retriever.retrieve(query, k=k)
        context_str = "\n".join(contexts)
        return self.generator.generate(context=context_str, query=query)

# --- Setup Functions ---

def load_data_and_build_index(csv_path):
    # Load the CSV
    data = pd.read_csv(csv_path)

    # Extract embeddings
    embedding_columns = [col for col in data.columns if col.isdigit()]
    embeddings = data[embedding_columns].to_numpy()
    texts = data['preprocessed_text'].tolist()

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, texts

def initialize_pipeline(csv_path, openai_api_key):
    # Load data + index
    index, texts = load_data_and_build_index(csv_path)

    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    retriever = Retriever(index=index, texts=texts, embedding_model=embedding_model)
    generator = Generator(api_key=openai_api_key)

    # Create pipeline
    rag_pipeline = RAGPipeline(retriever=retriever, generator=generator)
    return rag_pipeline

# --- Main Execution Example ---

if __name__ == "__main__":
    # Initialize
    csv_path = 'places_with_embeddings.csv'
    openai_api_key = 'your-openai-api-key'

    rag = initialize_pipeline(csv_path, openai_api_key)

    # Example query
    query = "What are the best cozy cafes in Brooklyn?"
    answer = rag.run(query, k=3)

    print("Answer:", answer)
