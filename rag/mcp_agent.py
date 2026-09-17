import asyncio

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic import BaseModel, Field, create_model

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()


# ============================================================
# RAG State
# ============================================================


class RAGState(MessagesState):
    patient_id: str


# ============================================================
# MCP Server Configuration
# ============================================================

server_params = StdioServerParameters(
    command="uv",
    args=["run", "python", "mcp_tool.py"],
)


# ============================================================
# Convert MCP JSON Schema → Pydantic Model
# ============================================================


def create_pydantic_model(tool_name, input_schema):
    """
    Convert the MCP tool inputSchema into a Pydantic model
    that LangChain can use for structured tool calling.
    """

    properties = input_schema.get("properties", {})
    required_fields = input_schema.get("required", [])

    fields = {}

    for field_name, field_info in properties.items():

        field_type = field_info.get("type", "string")
        description = field_info.get("description", "")

        # Map JSON Schema types to Python types
        if field_type == "string":
            python_type = str

        elif field_type == "integer":
            python_type = int

        elif field_type == "number":
            python_type = float

        elif field_type == "boolean":
            python_type = bool

        elif field_type == "array":
            python_type = list

        elif field_type == "object":
            python_type = dict

        else:
            python_type = str

        # Required fields
        if field_name in required_fields:

            fields[field_name] = (python_type, Field(..., description=description))

        # Optional fields
        else:

            fields[field_name] = (
                python_type | None,
                Field(default=None, description=description),
            )

    model = create_model(f"{tool_name}Input", **fields)

    return model


# ============================================================
# Load MCP Tools into LangChain
# ============================================================


async def load_mcp_tools(session):

    response = await session.list_tools()

    langchain_tools = []

    for mcp_tool in response.tools:

        tool_name = mcp_tool.name
        tool_description = mcp_tool.description or ""

        # Get the MCP tool's real input schema
        input_schema = mcp_tool.inputSchema

        # Convert MCP schema into Pydantic model
        args_schema = create_pydantic_model(tool_name, input_schema)

        # Create the MCP → LangChain bridge
        async def call_mcp_tool(tool_name=tool_name, **kwargs):

            # Remove None values
            arguments = {
                key: value for key, value in kwargs.items() if value is not None
            }

            result = await session.call_tool(tool_name, arguments=arguments)

            output = []

            for content in result.content:

                if hasattr(content, "text"):
                    output.append(content.text)

            return "\n".join(output)

        # Create structured LangChain tool
        tool = StructuredTool.from_function(
            coroutine=call_mcp_tool,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
        )

        langchain_tools.append(tool)

    return langchain_tools


# ============================================================
# Main Agent Function
# ============================================================


async def run_agent(
    question: str,
    patient_id: str,
):

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # ------------------------------------------------
            # Connect to MCP server
            # ------------------------------------------------

            await session.initialize()

            print("\nConnected to CLINSIGHT MCP Server")

            # ------------------------------------------------
            # Load MCP tools
            # ------------------------------------------------

            tools = await load_mcp_tools(session)

            print("\nMCP tools loaded into LangGraph:")

            for tool in tools:

                print(f"  - {tool.name}")

                print(f"    schema: {tool.args_schema.model_json_schema()}")

            # ------------------------------------------------
            # Create LLM
            # ------------------------------------------------

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            )

            llm_with_tools = llm.bind_tools(tools)

            # ------------------------------------------------
            # Agent Node
            # ------------------------------------------------

            def agent_node(state):

                system_message = SystemMessage(content="""
You are the CLINSIGHT healthcare AI assistant.

You have access to tools provided through the CLINSIGHT MCP server.

Use:

- search_patient_history for patient-specific clinical history.
- search_pubmed for medical research and scientific literature.
- summarize_pubmed for summarizing medical research.

Important rules:

- Use patient-specific information only when it comes from the patient history tool.
- Use PubMed for general medical literature.
- Keep patient information separate from general medical research.
- Do not invent patient information.
- Use tools when they are needed to answer the question.
- When the user provides a patient ID, use that patient ID when searching patient history.
""")

                messages = [system_message] + state["messages"]

                response = llm_with_tools.invoke(messages)

                print("\n--- AGENT NODE ---")
                print("Agent response generated")

                if response.tool_calls:

                    print("Agent requested tool(s):")

                    for tool_call in response.tool_calls:

                        print(f"  - {tool_call['name']}")

                        print(f"    arguments: {tool_call['args']}")

                else:

                    print("Agent produced final answer")

                return {"messages": [response]}

            # ------------------------------------------------
            # Decide whether to continue to tools
            # ------------------------------------------------

            def should_continue(state):

                last_message = state["messages"][-1]

                if last_message.tool_calls:

                    return "tools"

                return END

            # ------------------------------------------------
            # Build LangGraph
            # ------------------------------------------------

            graph_builder = StateGraph(RAGState)

            graph_builder.add_node("agent", agent_node)

            graph_builder.add_node("tools", ToolNode(tools))

            graph_builder.add_edge(START, "agent")

            graph_builder.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    END: END,
                },
            )

            graph_builder.add_edge("tools", "agent")

            graph = graph_builder.compile()

            # ------------------------------------------------
            # Run Agent
            # ------------------------------------------------

            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=f"""
                    Patient ID: {patient_id}

                    User question:
                    {question}
                            """)],
                    "patient_id": patient_id,
                }
                )

            # ------------------------------------------------
            # Get Final Answer
            # ------------------------------------------------

            final_message = result["messages"][-1]

            return final_message.content


# ============================================================
# Direct Test
# ============================================================


async def main():

    print("\nStarting CLINSIGHT MCP + LangGraph Agent...")

    print("============================================")

    question = (
        "What respiratory problems has P002 experienced, "
        "and what does recent medical research say "
        "about asthma exacerbation?"
    )

    answer = await run_agent(
        question=question,
        patient_id="P002",
    )

    print("\n==============================")

    print("FINAL ANSWER")

    print("==============================")

    print(answer)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
