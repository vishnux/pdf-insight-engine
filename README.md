<h1 align="center">PDF Insight Engine</h1>

<p align="center">
  Efficiently extract insights and provide answers from PDFs with low latency.
</p>

https://github.com/user-attachments/assets/c7847e81-d8a7-49c6-933c-38dbdbef6df0

# 1. Technical Documentation: System Architecture and Components
Overview:
This RAG system processes multilingual PDFs, extracting information and generating summaries and answers. It handles both scanned and digital PDFs using OCR and standard text extraction techniques.

Components:

Text Extraction:
OCR for Scanned PDFs: Integrated using Tesseract (extendable).
Digital PDF Parsing: Extracted using SimpleDirectoryReader.

Embedding Model: Utilizes HuggingFace’s "BAAI/bge-small-en-v1.5" for generating text embeddings.

VectorStore:
Stores document embeddings for efficient retrieval during query processing.

Query Decomposition:
Decomposes complex queries using a custom query engine based on a chat template.

Hybrid Search:
Combines keyword-based and semantic search techniques to retrieve relevant information.

Reranking Algorithm:
Ensures relevant answers are returned by reranking results based on semantic relevance.

System Architecture:

PDF Ingestion: PDFs are uploaded, analyzed, and stored.
Text Embedding: Text extracted from PDFs is embedded using HuggingFace models and stored in a vector database.
Query Processing: User queries are matched to the embeddings and reranked for relevance.
Answer Generation: A custom query template generates answers while retaining chat history.

# 2. User Guide
Instructions to Use:
Upload a PDF:

Open the sidebar and upload a PDF document (scanned or digital).
Process the Document:

Click “Process Document” to analyze the uploaded file.
Ask a Question:

Type your question in the input box and press enter.
The system retrieves relevant content from the document and generates a response.

System Maintenance:
Model Updates: Models can be updated or swapped via the HuggingFaceInferenceAPI and HuggingFaceEmbedding configurations.

# 3. Performance and Evaluation Report

Query Relevance:
The system retrieves relevant chunks of text from PDFs using a hybrid search mechanism, ensuring accurate responses.

Retrieval Test:
The system consistently returns document chunks aligned with the context of the user query.

Latency:
Average Latency: ~3-10s per query.

Fluency:
Generated Text: Responses are clear and coherent, enhanced by the reranking algorithm for improved readability.

Model Size:

Embedding Models: Small models such as BGE-small deliver fast and relevant results while maintaining response quality.

LLM: The "Gemma-1.1-7B" LLM strikes a balance between speed and accuracy.
