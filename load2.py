import json
import os
from dotenv import load_dotenv

import pinecone
from langchain.embeddings import OpenAIEmbeddings

# Load API keys and config
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME", "yc-startups-index")

# Step 1: Load the JSON data
with open("yc_startups.json", "r") as f:
    data = json.load(f)

descriptions = [entry["description"] for entry in data]

# Step 2: Generate embeddings
embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectors = embedder.embed_documents(descriptions)

# Step 3: Connect to Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

if index_name not in pinecone.list_indexes():
    raise ValueError(f"Index '{index_name}' not found in Pinecone. Create it first.")

index = pinecone.Index(index_name)

# Step 4: Format and upsert
records = [
    {
        "id": f"vector-{i}",
        "values": vectors[i],
        "metadata": {
            "name": entry["name"],
            "tagline": entry["tagline"],
            "industry": entry["industry"]
        }
    }
    for i, entry in enumerate(data)
]

# Optional: Chunk if you have 100+ entries
batch_size = 100
for i in range(0, len(records), batch_size):
    chunk = records[i:i+batch_size]
    index.upsert(vectors=chunk)

print(f"✅ Successfully upserted {len(records)} records into Pinecone index '{index_name}'.")
