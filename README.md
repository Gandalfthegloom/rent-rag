# RAG Q&A for Toronto Rent & Insurance

This project is a simple **Retrieval-Augmented Generation (RAG)** pipeline using LangChain. It lets you ask questions about Toronto rent and insurance laws, with answers based on a local set of PDF documents.

Everything runs **locally on your machine** — no cloud APIs required.

---

## 📦 Requirements

1. **Python 3.9+** (check with `python --version`).
2. Install required packages:

```bash
pip install -r requirements.txt
```

Recommended packages in `requirements.txt`:
- `langchain`
- `langchain-community`
- `langchain-chroma`
- `langchain-huggingface`
- `chromadb`
- `pypdf`
- `pygpt4all`

*(You may add more depending on your chosen local LLM integration.)*

3. **PDFs**: Place your source documents (Toronto rent & insurance PDFs) in the `data/` folder.

---

## ▶️ Running the Pipeline

Once requirements are installed:

```bash
python rag-pipeline.py
```

This will:
1. Load and split the PDFs.
2. Embed the text chunks.
3. Store them in a local vector database (Chroma).
4. Use GPT4All (or another local LLM) to answer your questions.

Example interaction:
```text
Question: What is rent control in Toronto?
Answer: Rent control limits the amount a landlord can increase rent annually... (etc)
```

---

## ⚙️ Notes
- On Windows, Ollama is not supported. Instead, this setup uses **GPT4All** for local LLM inference.
- For better performance, use small quantized models (e.g. `mistral-7b-instruct.Q4_0.gguf` or `tinyllama`).
- You can swap out the embedding model (`all-MiniLM-L6-v2`) if you want stronger embeddings.

---

## 📖 Project Structure
```
project/
│── data/                  # Store your PDFs here
│── main.py                # The main RAG pipeline
│── requirements.txt       # Python dependencies
│── README.md              # This file
```

