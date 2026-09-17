import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg2 import register_vector

load_dotenv()


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "proem_ai_rag",
    "user": "anusmacbook",
    "password": "",
}


def retrieve_patient_history(
    question: str,
    patient_id: str,
    top_k: int = 3,
):
    # print("STEP 1: Connecting to PostgreSQL", flush=True)

    conn = psycopg2.connect(**DB_CONFIG)

    # print("STEP 2: PostgreSQL connection successful", flush=True)

    register_vector(conn)

    # print("STEP 3: PGVector registered", flush=True)

    cursor = conn.cursor()

    # print("STEP 4: Cursor created", flush=True)

    # Create embedding for the user's question
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # print("STEP 5: Embedding model created", flush=True)

    query_embedding = embeddings.embed_query(question)

    query_vector = "[" + ",".join(map(str, query_embedding)) + "]"

    # print("Query vector type:", type(query_vector), flush=True)
    # print("Query vector dimensions:", len(query_embedding), flush=True)

    # print("STEP 6: Query embedding created", flush=True)
    # print("Type:", type(query_embedding), flush=True)
    # print("Length:", len(query_embedding), flush=True)
    # print("First 10 values:", query_embedding[:10], flush=True)

    # print("STEP 7: About to execute SQL", flush=True)

    # Search for the most similar chunks
    cursor.execute(
        """
        SELECT
            chunk_id,
            document_id,
            patient_id,
            content,
            embedding <=> %s::vector AS distance
        FROM document_chunks
        WHERE patient_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (
            query_vector,
            patient_id,
            query_vector,
            top_k,
        ),
    )

    # print("STEP 8: SQL executed successfully", flush=True)

    results = cursor.fetchall()

    # print(f"STEP 9: Retrieved {len(results)} chunks", flush=True)

    cursor.close()
    conn.close()

    return results


if __name__ == "__main__":

    question = "What respiratory problems has this patient experienced?"

    print("Starting retriever...", flush=True)

    results = retrieve_patient_history(
        question=question,
        patient_id="P002",
        top_k=3,
    )

    print("\nRetrieved chunks:\n")

    for chunk_id, document_id, patient_id, content, distance in results:
        print(f"Chunk ID: {chunk_id}")
        print(f"Patient: {patient_id}")
        print(f"Distance: {distance}")
        print(f"Content: {content}")
        print("-" * 60)