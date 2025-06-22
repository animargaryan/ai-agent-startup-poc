import os

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec, Pinecone

# Initialize a client.
# API key is required, but the value does not matter.
# Host and port of the Pinecone Local instance
# is required when starting without indexes.
from models import YCStartup

pc = PineconeGRPC(
    api_key=os.getenv("PINECONE_API_KEY"),
    host="http://localhost:5080"
)
dense_index_name = "startup-blogs"
namespace = "startup_ideas"


@tool
def retrieve_relevant_startups(description: str) -> list[YCStartup]:
    """Call out vector DB to find best matches for provided startup description.

    Args:
        description: Description of start up.

    Returns:
        List of relevant start ups.
    """
    dense_index_host = pc.describe_index(name=dense_index_name).host
    dense_index = pc.Index(host=dense_index_host, grpc_config=GRPCClientConfig(secure=False))
    embedder = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    description_vector = embedder.embed_query(description)

    # Query the dense index with a metadata filter
    dense_response = dense_index.query(
        namespace=namespace,
        vector=description_vector,
        # filter={"genre": {"$eq": "documentary"}},
        top_k=1,
        include_values=False,
        include_metadata=True
    )
    return [
        YCStartup(id=resp["id"], description=resp["metadata"]["description"], tagline=resp["metadata"]["tagline"],
                  industry=resp["metadata"]["industry"], name=resp["metadata"]["name"])
        for resp in dense_response["matches"]
    ]


def run_tool_call(res):
    print("🛠️ Tool call(s):", res.tool_calls)

    if not res.tool_calls:
        return {"possible_competitor": "NO TOOL CALL"}

    competitors = []

    for call in res.tool_calls:
        if call["name"] == "retrieve_relevant_startups":
            description = call["args"]["description"]
            # Actually call the tool function now
            result = retrieve_relevant_startups.invoke({"description": description})
            competitors.append(result)

    return {"possible_competitor": competitors}


if __name__ == "__main__":
    test_description = "An e-learning marketplace that connects learners in developing countries with volunteer mentors and curated open-source educational content, specifically tailored for low-bandwidth environments and mobile-first access. Focuses on foundational literacy, digital skills, and vocational training"
    response = retrieve_relevant_startups(test_description)
    print(response)