import logging
import os

# Bonus: stops a common warning about threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import OllamaLLM
import numpy as np
from dotenv import load_dotenv
import chromadb
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Load the document
loader = PyPDFLoader("./docs/Agentic_Design_Patterns.pdf")
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)

# Step 2: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
embeddings = model.encode([chunk.page_content for chunk in chunks])

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Step 3: Create ChromaDB collection and add documents and embeddings
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")
collection.add(
    documents=[chunk.page_content for chunk in chunks],
    embeddings=embeddings,
    ids = [str(uuid4()) for _ in chunks]
)

# Step 4: Retrieval function
def retrieve(query, k=2):
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding, 
        n_results=k
    )

    return results['documents'][0]

def rerank(query, documents, top_n=3):
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)
    
    # Sortiraj po score-u (opadajuće) i uzmi top_n
    return [documents[i] for i in np.argsort(scores)[-top_n:]]

# Step 5: RAG function
def rag_query(question):
    # Retrieve relevant docs
    candidates = retrieve(question, 10)

     # 2. Reranking kandidata
    context = rerank(question, candidates, top_n=3)

    # Create prompt
    prompt = f"""You are an expert assistant. Your task is to provide a thorough and detailed answer to the question using ONLY the information provided in the context below.

        Instructions:
        - Analyze the context carefully before answering
        - Provide a comprehensive, well-structured answer
        - Use concrete examples or details from the context to support your answer
        - If the context contains multiple relevant points, address all of them
        - If the context does not contain enough information to answer fully, explicitly state what is and isn't covered
        - Do NOT use any outside knowledge — rely solely on the provided context

        Context:
        {chr(10).join(context)}

        Question: {question}

        Provide a detailed answer below, organized with clear structure if needed:
        Answer:
    """

    # Generate response
    llm = OllamaLLM(
        model="gemma4:latest",
        temperature=0,
    )
    
    response = llm.invoke([{"role": "user", "content": prompt}])

    return response

def chat_loop():
    print("\nChatbot Started!")
    print("Type your queries or 'quit' to exit.")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
    
            if query.lower() == 'quit':
                break
            
            result = rag_query(query)
            print(f"\nAnswer: {result}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")