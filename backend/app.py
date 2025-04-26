from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.docstore import InMemoryDocstore

import pandas as pd
import numpy as np
import faiss

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- FAISS & Embedding Setup ---

# Read the CSV file and extract embeddings and texts
data = pd.read_csv('datatesting\\files\\p2\\datathon_p2\\Bruv.csv')
embedding_columns = [col for col in data.columns if col.isdigit()]
embeddings = data[embedding_columns].to_numpy()
texts = data['name'].tolist()
documents = [Document(page_content=text) for text in texts]

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Create the docstore and map FAISS indices to document IDs
docstore = InMemoryDocstore(dict(enumerate(documents)))

# Mapping FAISS index to docstore ID (IDs should be integers, not strings)
index_to_docstore_id = {i: i for i in range(len(documents))}

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the FAISS vectorstore
vectorstore = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Set up retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(
    openai_api_key="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", 
    model_name="gpt-3.5-turbo-instruct"
)

# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- Flask Endpoints ---

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_place_ids = [doc.page_content for doc in retrieved_docs]
        print(f"Retrieved Place IDs: {retrieved_place_ids}")

        # Get the answer from the QA chain
        answer = qa_chain.run(query)


        print(f"Answer: {answer}")

        return jsonify({
            "answer": answer,
            "retrieved_place_ids": retrieved_place_ids
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
