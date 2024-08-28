# from tqdm import tqdm
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# import os


# class MultiFileRAGTool:
#     def __init__(self, directory_path, groq_api_key=None, groq_model="llama3-8b-8192"):
#         self.directory_path = directory_path
#         self.groq_model = groq_model
#         self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
#         if not self.groq_api_key:
#             raise ValueError(
#                 "Groq API key must be provided or set as GROQ_API_KEY environment variable"
#             )
#         self.vectorstore = None
#         self.qa_chain = None
#         self.setup()

#     def setup(self):
#         print("Loading files...")
#         documents = self.load_documents()

#         print("Splitting text...")
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200
#         )
#         splits = text_splitter.split_documents(documents)

#         print("Creating embeddings and vector store...")
#         self.create_embeddings_and_vectorstore(splits)

#         print("Creating retriever and QA chain...")
#         retriever = self.vectorstore.as_retriever()

#         llm = ChatGroq(api_key=self.groq_api_key, model_name=self.groq_model)

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=llm, chain_type="stuff", retriever=retriever
#         )
#         print("Setup complete.")

#     def load_documents(self):
#         documents = []
#         files = os.listdir(self.directory_path)

#         # Use tqdm to add a progress bar for loading documents
#         for filename in tqdm(files, desc="Loading documents"):
#             file_path = os.path.join(self.directory_path, filename)
#             if filename.endswith(".pdf"):
#                 loader = PyPDFLoader(file_path)
#             elif filename.endswith(".txt"):
#                 loader = TextLoader(file_path)
#             elif filename.endswith(".docx"):
#                 loader = Docx2txtLoader(file_path)
#             else:
#                 print(f"Skipping unsupported file type: {filename}")
#                 continue
#             # Load and split each document, then extend the documents list
#             documents.extend(loader.load_and_split())
#         return documents

#     def create_embeddings_and_vectorstore(self, splits):
#         embedding_function = OllamaEmbeddings(model="all-minilm:latest")

#         # Specify the directory to store the Chroma vector store
#         persist_directory = r"C:\Users\HP\Desktop\crew ai\project2\db"
#         os.makedirs(persist_directory, exist_ok=True)

#         # Create the vector store and pass the embedding function
#         self.vectorstore = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embedding_function,  # Pass the embedding function directly
#         )

#         # Add texts to the vector store
#         for split in tqdm(splits, desc="Creating embeddings and vector store"):
#             self.vectorstore.add_texts(texts=[split.page_content])

#     def query(self, query: str) -> str:
#         """
#         Query the documents using the RAG system with Groq LLM.

#         Parameters:
#         - query (str): The question to ask about the content.

#         Returns:
#         - str: The answer to the query based on the content.
#         """
#         if not self.qa_chain:
#             raise ValueError("RAG system not initialized. Call setup() first.")

#         try:
#             result = self.qa_chain.invoke({"query": query})
#             return (
#                 result["result"]
#                 if isinstance(result, dict) and "result" in result
#                 else str(result)
#             )
#         except Exception as e:
#             return f"An error occurred while processing the query: {str(e)}"


# # Example usage
# if __name__ == "__main__":
#     rag_tool = MultiFileRAGTool(
#         r"C:\Users\HP\Desktop\crew ai\project2\ragfiles",
#         groq_api_key="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",
#     )
#     result = rag_tool.query(
#         "who is great king of chola and what is Personal Life of Napoleon Bonaparte?"
#     )
#     print(result)

# import os
# from tqdm import tqdm
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA


# def setup(directory_path, groq_api_key=None, groq_model="llama3-8b-8192"):
#     """
#     Set up the RAG system by loading documents, creating embeddings, and setting up the QA chain.

#     Parameters:
#     - directory_path: The path to the directory containing the documents.
#     - groq_api_key: The API key for Groq.
#     - groq_model: The Groq model to use.

#     Returns:
#     - A tuple containing the vectorstore and the QA chain.
#     """
#     if not groq_api_key:
#         raise ValueError("Groq API key must be provided")

#     print("Loading files...")
#     documents = load_documents(directory_path)

#     print("Splitting text...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(documents)

#     print("Creating embeddings and vector store...")
#     vectorstore = create_embeddings_and_vectorstore(splits)

#     print("Creating retriever and QA chain...")
#     retriever = vectorstore.as_retriever()
#     llm = ChatGroq(api_key=groq_api_key, model_name=groq_model)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm, chain_type="stuff", retriever=retriever
#     )

#     print("Setup complete.")
#     return vectorstore, qa_chain


# def load_documents(directory_path):
#     """
#     Load and split documents from a specified directory.

#     Parameters:
#     - directory_path: The path to the directory containing the documents.

#     Returns:
#     - A list of documents loaded from the directory.
#     """
#     documents = []
#     files = os.listdir(directory_path)

#     for filename in tqdm(files, desc="Loading documents"):
#         file_path = os.path.join(directory_path, filename)
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif filename.endswith(".txt"):
#             loader = TextLoader(file_path)
#         elif filename.endswith(".docx"):
#             loader = Docx2txtLoader(file_path)
#         else:
#             print(f"Skipping unsupported file type: {filename}")
#             continue
#         documents.extend(loader.load_and_split())
#     return documents


# def create_embeddings_and_vectorstore(splits):
#     """
#     Create embeddings and store them in a vector store.

#     Parameters:
#     - splits: The split documents to be processed.

#     Returns:
#     - The vector store containing the embeddings.
#     """
#     embedding_function = OllamaEmbeddings(model="all-minilm:latest")
#     persist_directory = r"C:\Users\HP\Desktop\crew ai\project2\db"
#     os.makedirs(persist_directory, exist_ok=True)

#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embedding_function,
#     )

#     for split in tqdm(splits, desc="Creating embeddings and vector store"):
#         vectorstore.add_texts(texts=[split.page_content])

#     return vectorstore


# def query(vectorstore, qa_chain, query_text):
#     """
#     Query the documents using the RAG system with Groq LLM.

#     Parameters:
#     - vectorstore: The vector store to use for retrieval.
#     - qa_chain: The QA chain to use for querying.
#     - query_text: The question to ask about the content.

#     Returns:
#     - The answer to the query based on the content.
#     """
#     if not qa_chain:
#         raise ValueError("RAG system not initialized. Call setup() first.")

#     try:
#         result = qa_chain.invoke({"query": query_text})
#         return (
#             result["result"]
#             if isinstance(result, dict) and "result" in result
#             else str(result)
#         )
#     except Exception as e:
#         return f"An error occurred while processing the query: {str(e)}"


# # Example usage
# if __name__ == "__main__":
#     vectorstore, qa_chain = setup(
#         r"C:\Users\HP\Desktop\crew ai\project2\ragfiles",
#         groq_api_key="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",
#     )
#     result = query(
#         vectorstore,
#         qa_chain,
#         "who is great king of chola and what is Personal Life of Napoleon Bonaparte and what is early life of Constantine the great?",
#     )
#     print(result)

import os
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from crewai_tools import tool
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
directory_path = r"C:\Users\HP\Desktop\crew ai\project2\ragfiles"


@tool
def setup_rag_system(
    directory_path=r"C:\Users\HP\Desktop\crew ai\project2\ragfiles",
    groq_api_key="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",
    groq_model="llama3-8b-8192",
):
    """
    Set up the RAG (Retrieval-Augmented Generation) system by loading documents,
    creating embeddings, and setting up the QA chain.

    Parameters:
    - directory_path: The path to the directory containing the documents.
    - groq_api_key: The API key for Groq.
    - groq_model: The Groq model to use (default is "llama3-8b-8192").

    Returns:
    - A string message indicating the setup completion.
    """

    # Ensure the API key is provided
    if not groq_api_key:
        return "Groq API key must be provided"

    # Load and split documents
    documents = load_documents(directory_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    vectorstore = create_embeddings_and_vectorstore(splits)

    # Set up the retriever and QA chain
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(api_key=groq_api_key, model_name=groq_model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    # Store the objects globally for reuse
    global stored_vectorstore, stored_qa_chain
    stored_vectorstore = vectorstore
    stored_qa_chain = qa_chain

    return "RAG system setup complete."


def load_documents(directory_path):
    """
    Load and split documents from a specified directory.
    """

    documents = []
    files = os.listdir(directory_path)

    for filename in tqdm(files, desc="Loading documents"):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue
        documents.extend(loader.load_and_split())
    return documents


def create_embeddings_and_vectorstore(splits):
    """
    Create embeddings and store them in a vector store.
    """

    embedding_function = OllamaEmbeddings(model="all-minilm:latest")
    persist_directory = r"C:\Users\HP\Desktop\crew ai\project2\db"
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )

    for split in tqdm(splits, desc="Creating embeddings and vector store"):
        vectorstore.add_texts(texts=[split.page_content])

    return vectorstore


@tool
def query_rag_system(query_text: str):
    """
    Query the documents using the RAG system with Groq LLM.

    Parameters:
    - query_text: The question to ask about the content.

    Returns:
    - The answer to the query based on the content.
    """

    if "stored_qa_chain" not in globals() or "stored_vectorstore" not in globals():
        return "RAG system not initialized. Please run setup_rag_system() first."

    try:
        result = stored_qa_chain.invoke({"query": query_text})
        return (
            result["result"]
            if isinstance(result, dict) and "result" in result
            else str(result)
        )
    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"
