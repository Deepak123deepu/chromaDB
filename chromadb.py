import chromadb

client = chromadb.Client()
collection = client.create_collection("yt_demo")

collection.add(
    documents=["This is a document about cat", "This is a document about car"],
    metadatas=[{"category": "animal"}, {"category": "vehicle"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["vehicle"],
    n_results=1
)
print(results)


# Reading Files from a Folder
import os

def read_files_from_folder(folder_path):
    file_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                content = file.read()
                file_data.append({"file_name": file_name, "content": content})

    return file_data

folder_path = "pets"
file_data = read_files_from_folder(folder_path)

#Adding File Contents to ChromaDB

documents = []
metadatas = []
ids = []

for index, data in enumerate(file_data):
    documents.append(data['content'])
    metadatas.append({'source': data['file_name']})
    ids.append(str(index + 1))

pet_collection = client.create_collection("pet_collection")

pet_collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

results = pet_collection.query(
    query_texts=["What are the different kinds of pets people commonly own?"],
    n_results=1
)
print(results)

# filtering result 

pet_collection.query(
    query_texts=["What are the emotional benefits of owning a pet?"],
    n_results=1,
    where_document={"$contains":"reptiles"}
)
print(results)


results = pet_collection.query(
    query_texts=["What are the emotional benefits of owning a pet?"],
    n_results=1,
    where={"source": "Training and Behaviour of Pets.txt"}
)
print(results)

#Using a different model for embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

documents = []
embeddings = []
metadatas = []
ids = []

for index, data in enumerate(file_data):
    documents.append(data['content'])
    embedding = model.encode(data['content']).tolist()
    embeddings.append(embedding)
    metadatas.append({'source': data['file_name']})
    ids.append(str(index + 1))

pet_collection_emb = client.create_collection("pet_collection_emb")

pet_collection_emb.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

query = "What are the different kinds of pets people commonly own?"
input_em = model.encode(query).tolist()

results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=1
)
print(results)

query = "foods that are recommended for dogs?"
input_em = model.encode(query).tolist()

results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=1
)
print(results)
