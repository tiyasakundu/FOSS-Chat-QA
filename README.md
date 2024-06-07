# Demo of a RAG based model for FOSSology Wiki

A RAG System which can load MD files (fossology wiki used here), embed them into a chromadb database, and responding to the queries given by a user.

1. RAG integrated with Langchain.
2. Fossology wiki provided as the external documents to the RAG, in md format.
3. Embeddings of these documents are created and stored in a database called ChromaDB.
4. User query is taken as input through CLI, and an optimized response is generated referring to the embedded database.

## Explanation
1. `create_database.py`: Outlines a process for loading, splitting, and saving text data from markdown files into a Chroma vector store.
   - The script loads markdown documents from a specified directory, splits these documents into manageable chunks, generates embeddings for these chunks using OpenAI's embedding model, and saves these embeddings into a Chroma vector store.
   - This process facilitates efficient storage and retrieval of document data, making it suitable for applications such as search engines or document analysis tools.

2. `query_data.py`: Querying a Chroma database to find relevant context for a given question and generating a response using an OpenAI model.
   - The script sets up a command-line interface to accept a question.
   - It initializes the necessary components for interacting with a Chroma vector store and an OpenAI model.
   - The script searches the vector store for relevant documents, formats a prompt with the found context, and queries the OpenAI model for an answer.
   - The final response, including sources of the context, is printed out.

3. `chromadb`: It is a vector database designed to handle large-scale, high-dimensional data efficiently. It is particularly suited for applications involving natural language processing, such as document search, question answering, recommendation systems, and more.

Integration with Langchain: LangChain is a framework designed to assist in building applications with large language models. ChromaDB can be integrated with LangChain to enhance its capabilities in managing and querying large datasets, making it a powerful combination for building intelligent applications. LangChain provides tools for document loading, text splitting, embedding generation, and querying, which can all be seamlessly integrated with ChromaDB's storage and retrieval capabilities.

## How to run
1. Clone the documents in `data/` folder and update the path in
   `create_database.py::DATA_PATH`.
    - For the demo, Wiki of FOSSology was cloned from repo
      `https://github.com/fossology/fossology.wiki.git`
2. Set the API key and base URL in `.env` for OpenAI like agents.
3. Run the `create_database.py` to load the `.md` files and create embedding
   database using ChromaDB.
4. Run the queries with `query_data.py "question"`