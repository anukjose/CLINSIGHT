import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_tool.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # -------------------------------------------------
            # 1. Initialize MCP connection
            # -------------------------------------------------

            await session.initialize()

            print("\nConnected to CLINSIGHT MCP Server")
            print("----------------------------------")

            # -------------------------------------------------
            # 2. Discover available tools
            # -------------------------------------------------

            response = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in response.tools:
                print(f"  - {tool.name}")

            # -------------------------------------------------
            # 3. Call patient history tool
            # -------------------------------------------------

            print("\nCalling search_patient_history...")
            print("----------------------------------")

            patient_result = await session.call_tool(
                "search_patient_history",
                arguments={
                    "question": "What respiratory problems has this patient experienced?",
                    "patient_id": "P002",
                },
            )

            print("\nPatient history result:")

            for content in patient_result.content:
                if hasattr(content, "text"):
                    print(content.text)

            # -------------------------------------------------
            # 4. Call PubMed tool
            # -------------------------------------------------

            print("\nCalling search_pubmed...")
            print("----------------------------------")

            pubmed_result = await session.call_tool(
                "search_pubmed",
                arguments={
                    "query": "asthma exacerbation",
                },
            )

            print("\nPubMed result:")

            for content in pubmed_result.content:
                if hasattr(content, "text"):
                    print(content.text)

            # -------------------------------------------------
            # 5. Finished
            # -------------------------------------------------

            print("\nMCP tool calls completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())