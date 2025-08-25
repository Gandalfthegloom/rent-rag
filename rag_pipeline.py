#############################################
# RAG Pipeline (Using Toronto Rent & Insurance as an example)
#############################################

# --- 1. Load and parse PDFs ---
from langchain_community.document_loaders import PyPDFLoader

# Load PDF into LangChain Document objects
pdf_path = "data/pdf/Standard Lease Guide - ENGLISH.pdf"  # <-- change to your PDF path
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages from {pdf_path}.")

# --- 2. Split text into chunks ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split text into ~500 characters with 50 overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# --- 3. Embed text chunks locally ---
from langchain_huggingface import HuggingFaceEmbeddings

# Lightweight embedding model (fast on CPU)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Test embedding
vec = embeddings.embed_query("What is rent control in Toronto?")
print(f"Example query vector length: {len(vec)}")

# --- 4. Store embeddings in Chroma (local vector DB) ---
from langchain_chroma import Chroma

# Create a persistent Chroma collection
vector_store = Chroma(
    collection_name="toronto_rent_insurance",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # saves to local disk
)

# Add documents (only run once; comment out to avoid duplication)
vector_store.add_documents(chunks)
print("Documents indexed in Chroma.")

# --- 5. Setup retriever ---
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 6. Connect to local LLM via GPT4All ---
from langchain_community.llms import GPT4All
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = GPT4All(
    model="mistral-7b-instruct-v0.1.Q4_0.gguf",    # Replace with your model path, if u already have one.
    allow_download=True,                                   # Let it fetch the model if missing
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True                                           # For debugging
)  

# --- 7. Create RetrievalQA chain ---
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# --- 8. Run a sample query ---
print("\nRAG Q&A system ready! Type your questions below (type 'exit' to quit).\n")
while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    if not query:
        continue
    try:
        answer = qa_chain.invoke({"query": query}, config={"callbacks": [StreamingStdOutCallbackHandler()]})
        print("Assistant:\n", answer['result'], "\n")
    except Exception as e:
        print("[Error]", e)

#############################################
# Notes:
# - Place your Toronto rent & insurance PDFs inside the `data/` folder.
# - The first run will embed and index the chunks. Later runs reuse ./chroma_db.
# - To test other questions, modify the `query` variable.
#############################################
