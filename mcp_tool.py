from mcp.server.fastmcp import FastMCP

from functions.symptom_extractor import extract_symptoms
from functions.diagnosis_symptoms import get_diagnosis
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.summerize_pubmed import summarize_text

from rag.retriever import retrieve_patient_history


mcp = FastMCP("Clinisight AI")


# ---------------------------------------------------------
# EXISTING MCP TOOL
# ---------------------------------------------------------

@mcp.tool()
async def clinisight_ai(symptom_text):
    symptom = extract_symptoms(symptom_text)
    diagnosis_result = get_diagnosis(symptom)
    pubmed_article = fetch_pubmed_articles_with_metadata(" ".join(symptom))
    summary = summarize_text(pubmed_article[:3000])

    return {
        "symptom": symptom,
        "diagnosis": diagnosis_result,
        "pubmed_summary": summary
    }


# ---------------------------------------------------------
# NEW MCP TOOL: PATIENT HISTORY
# ---------------------------------------------------------

@mcp.tool()
async def search_patient_history(
    question: str,
    patient_id: str
):
    """
    Search the patient's clinical history using vector similarity
    from PostgreSQL + pgvector.
    """

    results = retrieve_patient_history(
        question=question,
        patient_id=patient_id,
        top_k=3
    )

    if not results:
        return {
            "patient_id": patient_id,
            "results": [],
            "message": "No relevant patient history found."
        }

    formatted_results = []

    for result in results:
        formatted_results.append({
            "chunk_id": result[0],
            "document_id": result[1],
            "patient_id": result[2],
            "content": result[3],
            "distance": result[4]
        })

    return {
        "patient_id": patient_id,
        "results": formatted_results
    }

# ---------------------------------------------------------
# MCP TOOL: PUBMED SEARCH
# ---------------------------------------------------------

@mcp.tool()
async def search_pubmed(
    query: str
):
    """
    Search PubMed for medical research articles.
    """

    articles = fetch_pubmed_articles_with_metadata(
        query=query,
        max_results=3,
        use_mock_if_empty=False
    )

    if not articles:
        return {
            "query": query,
            "results": [],
            "message": "No PubMed articles found."
        }

    formatted_results = []

    for article in articles:
        formatted_results.append({
            "title": article.get("title"),
            "abstract": article.get("abstract"),
            "authors": article.get("authors"),
            "publication_date": article.get("publication_date"),
            "article_url": article.get("article_url")
        })

    return {
        "query": query,
        "results": formatted_results
    }
    
# ---------------------------------------------------------
# MCP TOOL: PUBMED SUMMARIZATION
# ---------------------------------------------------------

@mcp.tool()
async def summarize_pubmed(
    text: str
):
    """
    Summarize a medical research abstract or text.
    """

    if not text or not text.strip():
        return {
            "summary": "",
            "message": "No text was provided for summarization."
        }

    summary = summarize_text(text)

    return {
        "summary": summary
    }
    
# ---------------------------------------------------------
# RUN MCP SERVER
# ---------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")