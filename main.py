import os
import getpass
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# Set API key
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Initialize the reader and read the documents
reader = SimpleDirectoryReader(input_dir="./", required_exts=".csv")
documents = reader.load_data()

# Initialize the HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create the Chroma client and collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# Create the Chroma vector store with the defined collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Set chunk size and overlap
chunk_size = 200  # Adjust based on experimentation
chunk_overlap = 50  # Adjust based on experimentation

# Create the vector store index with the new embedding model
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)],
    embed_model=embed_model,
    storage_context=storage_context
)

# Save to disk
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# Take user input for the query
user_query = "What is the highest salary, who is it for, how old are they?"

# Query Data from the persisted index
query_engine = index.as_query_engine()
response = query_engine.query(user_query)

print(f"Response: {response}")
