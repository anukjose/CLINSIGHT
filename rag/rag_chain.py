from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retriever import retrieve_patient_history


def answer_patient_question(
    question: str,
    patient_id: str,
    top_k: int = 3,
):
    # 1. Retrieve relevant chunks
    results = retrieve_patient_history(
        question=question,
        patient_id=patient_id,
        top_k=top_k,
    )

    # 2. Convert retrieved chunks into context
    context = "\n\n".join(
        result[3]
        for result in results
    )

    # 3. Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a clinical information assistant.

Answer the user's question using only the patient
information provided in the context.

If the answer cannot be found in the context, say:
"I cannot find this information in the available patient history."

Do not invent or assume patient information.""",
            ),
            (
                "human",
                """Patient ID: {patient_id}

Context:
{context}

Question:
{question}""",
            ),
        ]
    )

    # 4. Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # 5. Create LangChain pipeline
    chain = prompt | llm

    # 6. Invoke the chain
    response = chain.invoke(
        {
            "patient_id": patient_id,
            "context": context,
            "question": question,
        }
    )

    return response.content


if __name__ == "__main__":

    question = "What respiratory problems has this patient experienced?"

    answer = answer_patient_question(
        question=question,
        patient_id="P002",
        top_k=3,
    )

    print("\nAnswer:\n")
    print(answer)