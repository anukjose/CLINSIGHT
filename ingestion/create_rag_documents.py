import psycopg2


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "proem_ai_rag",
    "user": "anusmacbook",
    "password": "",
}


def create_documents():

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            patient_id,
            visit_date,
            symptoms,
            diagnosis,
            notes
        FROM clinical_history
        ORDER BY patient_id, visit_date;
        """
    )

    records = cursor.fetchall()

    for patient_id, visit_date, symptoms, diagnosis, notes in records:

        content = f"""
Patient ID: {patient_id}
Clinical History Date: {visit_date}
Symptoms: {", ".join(symptoms)}
Diagnosis: {diagnosis}
Clinical Notes: {notes}
""".strip()

        cursor.execute(
            """
            INSERT INTO documents
            (patient_id, content)
            VALUES (%s, %s);
            """,
            (patient_id, content)
        )

    conn.commit()

    cursor.close()
    conn.close()

    print(f"Created {len(records)} RAG documents.")


if __name__ == "__main__":
    create_documents()