import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Define the request body for the query
class QueryRequest(BaseModel):
    query: str

# Absolute path to the Docs folder
docs_folder=os.path.join(os.getcwd(),'Docs')

# Filter out hidden / system files
valid_files=[file for file in os.listdir(docs_folder) if file.endswith(".pdf") and not file.endswith('.')]

# 1. Load PDFs from 'Docs/' folder
print("📂 Loading PDFs from 'Docs/'...")
loader = DirectoryLoader(
    docs_folder,              # <-- path is relative to this script
    glob="**/*.pdf",      # read all PDFs recursively
    loader_cls=PyPDFLoader
)

documents = loader.load()
if len(documents)==0:
    print(f" No documents found. Please check if the PDFs are in the correct folder and have .pdf extension.")
else:
    print(f"✅ Loaded {len(documents)} documents.")

# 2. Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Add file name as metadata to each chunk for source tracking
for i, chunk in enumerate(chunks):
    chunk.metadata['source'] = documents[i % len(documents)].metadata['source']

# 3. TF-IDF Vectorizer
corpus = [chunk.page_content for chunk in chunks]  # Accessing 'page_content' from chunk
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)
print("📊 TF-IDF vectorizer fitted.")

# 4. Ollama Embedding + FAISS index
model_name = "llama3.2:1b"   # Change to the correct model name if needed
embedding = OllamaEmbeddings(model=model_name)

try:
    print("🧠 Creating FAISS index with Ollama embeddings...")
    faiss_index = FAISS.from_documents(chunks, embedding)
    print("✅ FAISS index created.")
except Exception as e:
    print(f"❌ Failed to create FAISS index: {e}")
    faiss_index = None

# 5. Query handling and response generation
def get_response(query: str):
    query_tfidf = tfidf_vectorizer.transform([query])
    print("\n🔍 TF-IDF Vector for Query:")
    print(query_tfidf.toarray())

    # 6. Embed the query
    try:
        query_embedding = embedding.embed_query(query)
        print("\n🧠 Query embedding generated.")
    except Exception as e:
        print(f"❌ Error embedding query: {e}")
        query_embedding = None

    # 7. TF-IDF Search (Exact Match)
    print("\n🔍 Performing TF-IDF Search...")

    # Perform TF-IDF search (use the similarity score between query and corpus)
    tfidf_scores = (query_tfidf * tfidf_vectorizer.transform(corpus).T).toarray().flatten()

    # Check for NaN values and handle them
    tfidf_scores = np.nan_to_num(tfidf_scores)  # Convert NaN to zero if any

    # Normalize TF-IDF scores
    tfidf_scores_normalized = tfidf_scores / np.linalg.norm(tfidf_scores) if np.linalg.norm(tfidf_scores) != 0 else tfidf_scores

    # Ensure that the result is a valid score
    max_tfidf_score = np.max(tfidf_scores_normalized)

    # Check if TF-IDF score exceeds threshold (e.g., 0.2 for relevance)
    if max_tfidf_score > 0.2:  # Adjust threshold based on your needs
        best_tfidf_idx = np.argmax(tfidf_scores_normalized)
        response = chunks[best_tfidf_idx].page_content  # Show full content from the document chunk
        result = {"response": response, "source": chunks[best_tfidf_idx].metadata['source']}
        return result  # Return the most relevant content from the best match document
    else:
        # 8. FAISS Search and Scoring (Fallback to FAISS if no relevant match from TF-IDF)
        if faiss_index:
            print("\n🔎 Performing FAISS search...")
            results = faiss_index.similarity_search(query)

            # If no results found, show "I don't know" and skip showing any document details
            if not results:
                return {"response": "I don't know.Kindly input details related only to uploaded document... I am Special RAG for this model..."}  # Fallback response if no results are found
            else:
                # Filter the results to ensure the query is present in the document content (check for the exact query)
                filtered_results = [doc for doc in results if query.lower() in doc.page_content.lower()]

                if not filtered_results:
                    return {"response": "I don't know.Kindly input details related only to uploaded document... I am Special RAG for this model..."}  # No relevant content
                else:
                    response = filtered_results[0].page_content[:500]  # Show first 500 characters of the result
                    return {"response": response, "source": filtered_results[0].metadata['source']}  # Show the most relevant content

# 6. API endpoint for the query
@app.post("/query")
async def query_syllabus_high_school(query_request: QueryRequest):
    query = query_request.query
    result = get_response(query)
    return result