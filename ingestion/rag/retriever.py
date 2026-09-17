import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg2 import register_vector

load_dotenv()


# PostgreSQL connection
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
    print("STEP 1: Connecting to PostgreSQL")

    conn = psycopg2.connect(**DB_CONFIG)

    print("STEP 2: PostgreSQL connection successful")

    register_vector(conn)

    print("STEP 3: PGVector registered")

    cursor = conn.cursor()

    print("STEP 4: Cursor created")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    print("STEP 5: Embedding model created")

    query_embedding = embeddings.embed_query(question)

    print("STEP 6: Query embedding created")
    print("Type:", type(query_embedding))
    print("Length:", len(query_embedding))
    print("First 10:", query_embedding[:10])
    print("STEP 7: About to execute SQL")

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
            query_embedding,
            patient_id,
            query_embedding,
            top_k,
        ),
    )

    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


if __name__ == "__main__":

    question = "What respiratory problems has this patient experienced?"

    results = retrieve_patient_history(
        question=question,
        patient_id="P002",
        top_k=3,
    )

    print("\nRetrieved chunks:\n")

    for (
        chunk_id,
        document_id,
        patient_id,
        content,
        distance,
    ) in results:

        print(f"Chunk ID: {chunk_id}")
        print(f"Document ID: {document_id}")
        print(f"Patient: {patient_id}")
        print(f"Distance: {distance}")
        print(f"Content: {content}")
        print("-" * 60)