from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os




reader = SimpleDirectoryReader(input_dir="./", required_exts=".csv")
documents = reader.load_data()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=50, chunk_overlap=10)],
    embed_model=embed_model,
)

# Access and print the stored vectors and metadata
print("Vector Store Contents:")
for doc_id, doc in index.docstore.docs.items():
    print(f"Document ID: {doc_id}")
    print(f"Text: {doc.text}")
    print(f"Metadata: {doc.metadata}")
    print(f"Embedding: {doc.embedding}")
    print("\n")

# Access and print nodes
print("Stored Vectors:")
for node in index.docstore.docs.values():
    print(f"Document ID: {node.id_}")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}")
    print(f"Embedding: {node.embedding}")
    print("\n")