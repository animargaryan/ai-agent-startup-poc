from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI

# ---- LLM Setup ----
from retriever import retrieve_relevant_startups, run_tool_call

llm_with_tool = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0
).bind_tools([retrieve_relevant_startups])

# ---- Prompt Templates ----
prompt1 = PromptTemplate.from_template("""
Step1 :

I want to start something in {industry}. Could you generate 3 possible start up ideas? 

A:
""")

prompt2 = PromptTemplate.from_template("""
Step 2:

For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. 

{startup_ideas}

A:""")

prompt3 = PromptTemplate.from_template("""
Step 3:

Given these startup ideas, retrieve relevant competitors:

{startup_ideas}
A:""")

prompt4 = PromptTemplate.from_template("""
Step 4:

Based on the possible competitors list {possible_competitor} and your {review} rank the startup ideas and return the most promising one

A:""")

# ---- Step 1 ----
step1 = (
    prompt1
    | llm_with_tool
    | RunnableLambda(lambda res: {"startup_ideas": res.content})
)

# ---- Step 2 ----
step2 = (
    RunnableLambda(lambda x: {"startup_ideas": x["startup_ideas"]})
    | prompt2
    | llm_with_tool
    | RunnableLambda(lambda res: {"review": res.content})
)

# ---- Step 3 ----
step3 = (
    RunnableLambda(lambda x: {"startup_ideas": x["startup_ideas"]})
    | prompt3
    | llm_with_tool
    | RunnableLambda(run_tool_call)

)

# ---- Step 4 ----
step4 = (
    RunnableLambda(lambda x: {
        "review": x["review"],
        "possible_competitor": x["possible_competitor"]
    })
    | prompt4
    | llm_with_tool
)

# ---- Full Chain ----
full_chain = (
    RunnableLambda(lambda x: {"industry": x["industry"]})
    | step1
    | RunnableLambda(lambda x: {"startup_ideas": x["startup_ideas"]})
    | RunnableMap({
        "review": step2,
        "possible_competitor": step3
    })
    | step4
)


if __name__ == "__main__":
    # ---- Run ----
    result = full_chain.invoke({"industry": "education technology"})
    print(result.content)
