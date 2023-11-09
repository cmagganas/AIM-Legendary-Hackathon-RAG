import random
import time

import faker
import openai
import pinecone
import tqdm
from datasets import Dataset

fake = faker.Faker()
pinecone_index_name = "coherererank"
dimension = 1536
embed_model = "text-embedding-ada-002"


# In this function, we're setting up our connection to Pinecone, a vector database that helps us in storing and querying vectorized data.
def initialize_pinecone_index(api_key, env, pinecone_index_name, dimension):
    """
    Initializes a Pinecone index for similarity search.

    Args:
        api_key (str): The API key for accessing Pinecone.
        env (str): The environment for Pinecone.
        index_name (str): The name of the index to initialize.
        dimension (int): The dimension of the index.

    Returns:
        pinecone.Index: The initialized Pinecone index.
    """
    print("Initializing Pinecone...")
    pinecone.init(api_key=api_key, environment=env)
    if index_name not in pinecone.list_indexes():
        print(f"Creating Pinecone index: {index_name}")
        pinecone.create_index(index_name, dimension=dimension, metric="dotproduct")
        while not pinecone.describe_index(index_name).status["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(1)
    index = pinecone.Index(index_name)
    print("Pinecone initialized successfully!")
    return index


# Generates a synthetic resume by creating a dictionary with randomly generated data for fields such as name, job, company, skills, experience, and education.
def generate_synthetic_resume():
    """
    Generates a synthetic resume with various fields filled with random, but plausible data.

    Returns:
        dict: The generated synthetic resume.
    """
    print("Generating a synthetic resume...")
    resume = {
        "id": fake.uuid4(),
        "text": f"{fake.name()}\n{fake.job()}\n{fake.company()}\n{fake.catch_phrase()}\nSkills: {', '.join(fake.words(ext_word_list=None, unique=True))}\nExperience: {fake.bs()} at {fake.company()} for {random.randint(1, 10)} years.",
        "metadata": {
            "experience": f"{random.randint(1, 10)} years",
            "education": random.choice(["Bachelor's", "Master's", "PhD"]),
        },
    }
    print("Synthetic resume generated successfully!")
    return resume


# In this function, we are focusing on creating a dataset of synthetic resumes. This is particularly useful for simulating a real-world scenario where you have a collection of resumes to work with.
def create_synthetic_resumes_dataset(num_resumes=1000, chunk_size=800):
    """
    Creates a dataset of synthetic resumes.

    Args:
        num_resumes (int, optional): The number of synthetic resumes to generate. Defaults to 1000.
        chunk_size (int, optional): The size of each text chunk. Defaults to 800.

    Returns:
        datasets.Dataset: The created dataset of synthetic resumes.
    """
    print("Creating dataset...")
    synthetic_resumes = [generate_resume() for _ in range(num_resumes)]
    data = []
    for resume in synthetic_resumes:
        resume_text = resume["text"]
        text_chunks = [
            resume_text[i : i + chunk_size]
            for i in range(0, len(resume_text), chunk_size)
        ]
        for idx, chunk in enumerate(text_chunks):
            chunk_id = f'{resume["id"]}-{idx}'
            data_entry = {
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "title": "Resume Chunk",
                    "url": f"https://example.com/resume/{chunk_id}",
                    "primary_category": "Resume",
                    "published": "20231028",
                    "updated": "20231028",
                    "text": chunk,
                },
            }
            data.append(data_entry)
    dataset_dict = {
        "id": [item["id"] for item in data],
        "text": [item["text"] for item in data],
        "metadata": [item["metadata"] for item in data],
    }
    formatted_dataset = Dataset.from_dict(dataset_dict)
    print("Dataset created successfully!")
    return formatted_dataset


# This function is crucial for converting our text data into numerical vectors, which is a format that can be understood and processed by machine learning models.
def embed_documents(docs: list[str]) -> list[list[float]]:
    """
    Embeds a list of documents using an embedding model.

    Args:
        docs (list[str]): The list of documents to embed.

    Returns:
        list[list[float]]: The embeddings of the documents.
    """
    print("Embedding documents...")
    res = openai.Embedding.create(input=docs, engine=embed_model)
    print("Documents embedded successfully!")
    return [x["embedding"] for x in res["data"]]


# In this function, we are focused on inserting our dataset into the Pinecone index.
def insert_dataset_to_pinecone_index(pinecone_index, dataset, batch_size=100):
    """
    Inserts a dataset into the Pinecone index.

    Args:
        index: The Pinecone index to insert the data into.
        dataset: The dataset to be inserted.
        batch_size (int, optional): The size of each batch for insertion. Defaults to 100.
    rerank_response = cohere_client.rerank(
        query=query,
        documents=doc_texts,
        top_n=rerank_top_n,
        model="rerank-english-v2.0",
    )
    """
    response = cohere_client.generate(prompt=prompt)
    if response.generations:
        print("Resumes evaluated successfully!")
        return response.generations[0].text, None
    else:
        print("Failed to generate a response.")
        return None, "Failed to generate a response."

    Returns:
        str: The generated text containing the evaluations and justifications.
        str: An error message if the response generation fails.
    """
    index_stats = index.describe_index_stats()
    if index_stats.total_vector_count > 0:
        print("Pinecone index is not empty. No new data will be inserted.")
        return

    # Fetch existing vector IDs in the index
    response = index.fetch(ids=dataset["id"])
    existing_ids = set(response.get("id", []))

    # Filter out the data that is already in the index
    new_data = dataset.filter(lambda example: example["id"] not in existing_ids)

    if len(new_data) == 0:
        print("All data is already present in the Pinecone index.")
        return

    # Insert the new data in batches
    for i in range(0, len(new_data), batch_size):
        batch = new_data[i : i + batch_size]
        embeds = embed(batch["text"])
        to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
        index.upsert(vectors=to_upsert)
        print(
            f"Batch {i // batch_size + 1}/{(len(new_data) - 1) // batch_size + 1} inserted."
        )

    print("New data inserted to Pinecone successfully!")


# In this function, we are querying the Pinecone index to fetch documents that are most relevant to a given query.
def fetch_documents_from_pinecone_index(pinecone_index, query: str, top_k: int):
    """
    Fetches documents from a Pinecone index based on a query.

    Args:
        index: The Pinecone index to query.
        query (str): The query string.
        top_k (int): The number of top documents to retrieve.

    Returns:
        dict: A dictionary of fetched documents, with the text as the key and the original rank as the value.
    """
    print("Fetching documents from Pinecone...")
    xq = embed([query])[0]
    res = index.query(xq, top_k=top_k, include_metadata=True)
    docs = {x["metadata"]["text"]: i for i, x in enumerate(res["matches"])}
    print("Documents fetched successfully!")
    rerank_docs = [result.document for result in rerank_response.results]
    combined_resumes = "\n\n".join([doc["text"] for doc in rerank_docs])

    prompt = f"""
    You are an HR professional with extensive experience in evaluating resumes for various job roles.This is the task you have been assigned.
    Task:
    {query}
    Based on the resumes provided below, your task is to select the top candidates and provide a detailed justification for each selection, highlighting their skills, experience, and overall fit for a general job role. Focus solely on the evaluation and selection process, and ensure your response is clear, concise, and directly related to the task at hand.

    ---

    Resumes:
    {combined_resumes}

    ---

    Please provide your selections and detailed justifications below:
    """
    Evaluates resumes based on a given job query.

    Args:
        pinecone_index: The Pinecone index to perform the initial search.
        cohere_client: The Cohere reranking model.
        query (str): The job query.
        top_k (int, optional): The number of top resumes to retrieve from the initial search. Defaults to 10.
        rerank_top_n (int, optional): The number of resumes to consider after reranking. Defaults to 5.
    """
    print("Evaluating resumes...")
    docs = fetch_documents_from_pinecone_index(pinecone_index, query, top_k=top_k)
    if not docs:
        print("No documents found.")
        return None, "No documents found."
    doc_texts = list(docs.keys())
    response = cohere_client.generate(prompt=prompt)
    if response.generations:
        print("Resumes evaluated successfully!")
        return response.generations[0].text, None
    else:
        print("Failed to generate a response.")
        return None, "Failed to generate a response."

    Returns:
        str: The generated text containing the evaluations and justifications.
        str: An error message if the response generation fails.
    """
    rerank_response = cohere_client.rerank(
        query=query,
        documents=doc_texts,
        top_n=rerank_top_n,
        model="rerank-english-v2.0",
    )
        pinecone_index: The Pinecone index to perform the initial search.
        cohere_client: The Cohere reranking model.
        query (str): The job query.
        top_k (int, optional): The number of top resumes to retrieve from the initial search. Defaults to 10.
        rerank_top_n (int, optional): The number of resumes to consider after reranking. Defaults to 5.

    Returns:
        str: The generated text containing the evaluations and justifications.
        str: An error message if the response generation fails.
    print("Evaluating resumes...")
    docs = get_docs(index, query, top_k=top_k)
    if not docs:
        print("No documents found.")
        return None, "No documents found."
    doc_texts = list(docs.keys())
    rerank_response = co.rerank(
        query=query,
        documents=doc_texts,
        top_n=rerank_top_n,
        model="rerank-english-v2.0",
    )
    rerank_docs = [result.document for result in rerank_response.results]
    combined_resumes = "\n\n".join([doc["text"] for doc in rerank_docs])

    prompt = f"""
    You are an HR professional with extensive experience in evaluating resumes for various job roles.This is the task you have been assigned.
    Task:
    {query}
    Based on the resumes provided below, your task is to select the top candidates and provide a detailed justification for each selection, highlighting their skills, experience, and overall fit for a general job role. Focus solely on the evaluation and selection process, and ensure your response is clear, concise, and directly related to the task at hand.

    ---

    Resumes:
    {combined_resumes}

    ---

    Please provide your selections and detailed justifications below:
    """
    response = co.generate(prompt=prompt)
    if response.generations:
        print("Resumes evaluated successfully!")
        return response.generations[0].text, None
    else:
        print("Failed to generate a response.")
        return None, "Failed to generate a response."
    