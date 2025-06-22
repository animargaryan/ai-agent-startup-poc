import json
import os

from pinecone import ServerlessSpec
from pinecone.grpc import GRPCClientConfig, PineconeGRPC
from langchain_openai import OpenAIEmbeddings

from retriever import dense_index_name, namespace


def load_data_to_pinecone() :
    with open('yc_startups.json', 'r') as file:
        data = json.load(file)

    # Initialize a client.
    # API key is required, but the value does not matter.
    # Host and port of the Pinecone Local instance
    # is required when starting without indexes.
    pc = PineconeGRPC(
        api_key=os.getenv("PINECONE_API_KEY"),
        host="http://localhost:5080"
    )

    if pc.has_index(name=dense_index_name):
        pc.delete_index(name=dense_index_name)

    if not pc.has_index(name=dense_index_name):
        dense_index_model = pc.create_index(
            name=dense_index_name,
            vector_type="dense",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
            tags={"environment": "development"}
        )

        print("\nDense index model:\n", dense_index_model)

    # Target index, disabling tls
    dense_index_host = pc.describe_index(name=dense_index_name).host
    dense_index = pc.Index(host=dense_index_host, grpc_config=GRPCClientConfig(secure=False))

    # Convert the text into numerical vectors that Pinecone can index
    embedder = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = embedder.embed_documents([d['description'] for d in data])

    new_data = [
        {
            "id": f"vector{idx}",
            "values": emb,
            "metadata": {
                "name": entry["name"],
                "tagline": entry["tagline"],
                "industry": entry["industry"],
                "description": entry["description"]
            }
        }
        for idx, (entry, emb) in enumerate(zip(data, embeddings))
    ]

    dense_index.upsert(namespace=namespace, vectors=new_data)


if __name__ == "__main__":
    load_data_to_pinecone()
