import json
from pathlib import Path

import psycopg2


# PostgreSQL connection details
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "proem_ai_rag",
    "user": "anusmacbook",
    "password": "",
}


# Folder containing patient JSON files
DATA_DIR = Path("data")


def ingest_patient(patient_data):

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    patient_id = patient_data["patient_id"]

    # Insert patient
    cursor.execute(
        """
        INSERT INTO patients (patient_id, age, gender)
        VALUES (%s, %s, %s)
        ON CONFLICT (patient_id) DO NOTHING;
        """,
        (
            patient_id,
            patient_data["demographics"]["age"],
            patient_data["demographics"]["gender"],
        ),
    )

    # Insert medical conditions
    for condition in patient_data["medical_conditions"]:
        cursor.execute(
            """
            INSERT INTO conditions (patient_id, condition_name)
            VALUES (%s, %s);
            """,
            (patient_id, condition),
        )

    # Insert medications
    for medication in patient_data["medications"]:
        cursor.execute(
            """
            INSERT INTO medications
            (patient_id, name, dosage, frequency, start_date, status)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (
                patient_id,
                medication["name"],
                medication["dosage"],
                medication["frequency"],
                medication["start_date"],
                medication["status"],
            ),
        )

    # Insert prescriptions
    for prescription in patient_data["prescriptions"]:
        cursor.execute(
            """
            INSERT INTO prescriptions
            (
                prescription_id,
                patient_id,
                medication,
                dosage,
                frequency,
                prescribed_date,
                duration
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (prescription_id) DO NOTHING;
            """,
            (
                prescription["prescription_id"],
                patient_id,
                prescription["medication"],
                prescription["dosage"],
                prescription["frequency"],
                prescription["prescribed_date"],
                prescription["duration"],
            ),
        )

    # Insert clinical history
    for history in patient_data["clinical_history"]:
        cursor.execute(
            """
            INSERT INTO clinical_history
            (
                patient_id,
                visit_date,
                symptoms,
                diagnosis,
                notes
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                patient_id,
                history["date"],
                history["symptoms"],
                history["diagnosis"],
                history["notes"],
            ),
        )

    conn.commit()

    cursor.close()
    conn.close()

    print(f"Successfully ingested {patient_id}")


def main():

    json_files = list(DATA_DIR.glob("*.json"))

    print(f"Found {len(json_files)} patient files.")

    for file_path in json_files:

        print(f"Reading: {file_path}")

        with open(file_path, "r") as file:
            patient_data = json.load(file)

        ingest_patient(patient_data)


if __name__ == "__main__":
    main()