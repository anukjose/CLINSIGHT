from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from rag.retriever import retrieve_patient_history
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.summerize_pubmed import summarize_text


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. GRAPH STATE
# ============================================================

class RAGState(MessagesState):
    patient_id: str


# ============================================================
# 3. TOOL 1
# PATIENT HISTORY SEARCH
# ============================================================

@tool
def search_patient_history(
    question: str,
    patient_id: str,
) -> str:
    """
    Search the patient's clinical history using
    semantic vector search.

    Use this tool for patient-specific questions
    about symptoms, diagnoses, clinical notes,
    and previous medical history.
    """

    print(
        "\n--- TOOL 1: SEARCH PATIENT HISTORY ---",
        flush=True,
    )

    # Reuse our existing retriever
    results = retrieve_patient_history(
        question=question,
        patient_id=patient_id,
        top_k=3,
    )

    print(
        f"Retrieved {len(results)} chunks",
        flush=True,
    )

    if not results:
        return (
            "No relevant patient history "
            "was found."
        )

    output = []

    for (
        chunk_id,
        document_id,
        patient_id,
        content,
        distance,
    ) in results:

        output.append(
            f"""
Document ID: {document_id}
Chunk ID: {chunk_id}
Patient ID: {patient_id}
Distance: {distance}

Clinical Information:
{content}
"""
        )

    return "\n".join(output)


# ============================================================
# 4. TOOL 2
# PUBMED SEARCH
# ============================================================

@tool
def search_pubmed(
    query: str,
) -> str:
    """
    Search PubMed for medical research relevant
    to the user's question.

    Use this tool for medical literature,
    scientific studies, and research evidence.
    """

    print(
        "\n--- TOOL 2: SEARCH PUBMED ---",
        flush=True,
    )

    # Reuse our existing PubMed function
    articles = fetch_pubmed_articles_with_metadata(
        query=query,
        max_results=3,
        use_mock_if_empty=False,
    )

    print(
        f"Found {len(articles)} PubMed articles",
        flush=True,
    )

    if not articles:
        return (
            "No relevant PubMed articles "
            "were found."
        )

    output = []

    for article in articles:

        output.append(
            f"""
Title:
{article["title"]}

Authors:
{", ".join(article["authors"])}

Publication Date:
{article["publication_date"]}

Abstract:
{article["abstract"]}

PubMed URL:
{article["article_url"]}
"""
        )

    return "\n".join(output)


# ============================================================
# 5. TOOL 3
# PUBMED SUMMARIZATION
# ============================================================

@tool
def summarize_pubmed(
    text: str,
) -> str:
    """
    Summarize medical research text or
    PubMed abstracts.

    Use this tool when the user asks for
    a summary of research.
    """

    print(
        "\n--- TOOL 3: SUMMARIZE PUBMED ---",
        flush=True,
    )

    # Reuse our existing summarization function
    summary = summarize_text(text)

    return summary


# ============================================================
# 6. CREATE LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ============================================================
# 7. REGISTER TOOLS
# ============================================================

tools = [
    search_patient_history,
    search_pubmed,
    summarize_pubmed,
]


# Give all tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 8. AGENT NODE
# ============================================================

def agent_node(
    state: RAGState,
):

    print(
        "\n--- AGENT NODE ---",
        flush=True,
    )

    system_message = SystemMessage(
        content="""
You are a healthcare information assistant.

You have access to three tools.

==================================================
TOOL 1: search_patient_history
==================================================

Searches the patient's own clinical records
using semantic vector search.

Use this tool when the question is about
a specific patient's:

- symptoms
- diagnoses
- clinical history
- clinical notes
- previous medical problems


==================================================
TOOL 2: search_pubmed
==================================================

Searches PubMed for medical research.

Use this tool when the question asks about:

- medical research
- scientific evidence
- studies
- medical literature
- research findings
- general medical knowledge


==================================================
TOOL 3: summarize_pubmed
==================================================

Summarizes medical research text or abstracts.

Use this tool when the user asks to:

- summarize research
- summarize PubMed articles
- summarize medical abstracts


==================================================
TOOL SELECTION
==================================================

Choose the appropriate tool based on the
user's question.

You may use multiple tools if necessary.

For example:

Question:
"What respiratory problems has P002 experienced?"

Use:
search_patient_history


Question:
"What does research say about asthma exacerbation?"

Use:
search_pubmed


Question:
"Find research about asthma exacerbation
and summarize the findings."

Use:
search_pubmed
then summarize_pubmed


Question:
"What respiratory problems has P002 experienced,
and what does research say about asthma?"

Use:
search_patient_history
and search_pubmed.


==================================================
IMPORTANT RULES
==================================================

Use only information returned by the tools.

Do not invent patient information.

Do not make unsupported assumptions.

Keep patient-specific information separate
from general medical literature.

Do not treat medical literature as evidence
that a particular patient has a condition.

If the available information does not answer
the question, clearly say that the information
was not found.
"""
    )

    # Add system message to existing conversation
    messages = [
        system_message
    ] + state["messages"]

    # Ask the LLM what to do
    response = llm_with_tools.invoke(
        messages
    )

    print(
        "Agent response generated",
        flush=True,
    )

    # Check whether the agent requested tools
    if response.tool_calls:

        print(
            "Agent requested tool(s):",
            flush=True,
        )

        for tool_call in response.tool_calls:

            print(
                f"  - {tool_call['name']}",
                flush=True,
            )

            print(
                f"    arguments: {tool_call['args']}",
                flush=True,
            )

    else:

        print(
            "Agent produced final answer",
            flush=True,
        )

    # IMPORTANT:
    # MessagesState automatically appends this
    # message to the existing message history.
    return {
        "messages": [response]
    }


# ============================================================
# 9. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 10. ROUTING LOGIC
# ============================================================

def should_continue(
    state: RAGState,
):

    last_message = state["messages"][-1]

    # If the agent requested a tool,
    # send execution to ToolNode.
    if last_message.tool_calls:

        return "tools"

    # Otherwise the agent has produced
    # the final answer.
    return END


# ============================================================
# 11. BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    RAGState
)


# Add Agent node
graph_builder.add_node(
    "agent",
    agent_node,
)


# Add Tool node
graph_builder.add_node(
    "tools",
    tool_node,
)


# ============================================================
# 12. GRAPH EDGES
# ============================================================

# START
#   ↓
# AGENT

graph_builder.add_edge(
    START,
    "agent",
)


# AGENT
#   ↓
# TOOL
#
# OR
#
# AGENT
#   ↓
# END

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)


# TOOL
#   ↓
# AGENT

graph_builder.add_edge(
    "tools",
    "agent",
)


# ============================================================
# 13. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 14. TEST THE GRAPH
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Agentic RAG...",
        flush=True,
    )

    patient_id = "P002"

    question = (
        #"What respiratory problems has this patient experienced?"
        #"What does medical research say about asthma exacerbation?"
        "What respiratory problems has P002 experienced, and what does recent medical research say about asthma exacerbation?"
    )

    # Initial conversation
    initial_state = {

        "messages": [

            HumanMessage(
                content=(
                    f"Patient ID: {patient_id}\n\n"
                    f"Question: {question}"
                )
            )

        ],

        "patient_id": patient_id,
    }

    # Run LangGraph
    result = graph.invoke(
        initial_state
    )


    # ========================================================
    # 15. FINAL ANSWER
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "==============================\n"
    )

    print(
        result["messages"][-1].content
    )