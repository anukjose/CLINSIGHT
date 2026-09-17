from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# PostgreSQL connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "proem_ai_rag",
    "user": "anusmacbook",
    "password": "",
}


def create_embeddings():

    # 1. Connect to PostgreSQL
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 2. Get documents from PostgreSQL
    cursor.execute(
        """
        SELECT document_id, patient_id, content
        FROM documents
        ORDER BY document_id;
        """
    )

    documents = cursor.fetchall()

    print(f"Found {len(documents)} documents.")

    # 3. LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    # 4. OpenAI embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    total_chunks = 0

    # 5. Process each document
    for document_id, patient_id, content in documents:

        print(f"Processing document {document_id} for {patient_id}")

        chunks = text_splitter.create_documents([content])

        # 6. Generate embeddings for all chunks
        chunk_texts = [chunk.page_content for chunk in chunks]

        vectors = embeddings.embed_documents(chunk_texts)

        # 7. Store chunks + embeddings in PostgreSQL
        for chunk_index, (chunk_text, vector) in enumerate(
            zip(chunk_texts, vectors)
        ):

            cursor.execute(
                """
                INSERT INTO document_chunks
                (
                    document_id,
                    chunk_index,
                    patient_id,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    document_id,
                    chunk_index,
                    patient_id,
                    chunk_text,
                    vector,
                ),
            )

            total_chunks += 1

    conn.commit()

    cursor.close()
    conn.close()

    print(f"Created {total_chunks} chunks and embeddings.")


if __name__ == "__main__":
    create_embeddings()