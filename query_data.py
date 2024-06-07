import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    load_dotenv()
    #Create CLI
    parser = argparse.ArgumentParser(description="Query a Chroma database.")
    parser.add_argument("question", type=str, help="The question to ask the database.")
    args = parser.parse_args()
    question = args.question

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model="bge-m3",
        )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(question, k=3)

    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    print(prompt)

    model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model="mistral-7b-instruct",
        )
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
