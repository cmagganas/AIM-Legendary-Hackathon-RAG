import os

import cohere
import helpers
import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Return the Cohere client and Pinecone index
def initialize_apis():
    if "openai_api_key" in st.session_state and "cohere_api_key" in st.session_state:
        openai.api_key = st.session_state["openai_api_key"]
        co = cohere.Client(st.session_state["cohere_api_key"])
        index = helpers.initialize_pinecone(
            st.session_state["api_key"], st.session_state["env"], "coherererank", 1536
        )
        return co, index
    return None, None


# Streamlit Sidebar for API Key Input
with st.sidebar:
    api_key = st.text_input(
        "Enter Pinecone API key:", value=os.getenv("PINECONE_API_KEY", "")
    )
    env = st.text_input(
        "Enter Pinecone environment:", value=os.getenv("PINECONE_ENVIRONMENT", "")
    )
    openai_api_key = st.text_input(
        "Enter OpenAI API key:", value=os.getenv("OPENAI_API_KEY", "")
    )
    cohere_api_key = st.text_input(
        "Enter Cohere API key:", value=os.getenv("COHERE_API_KEY", "")
    )

    if st.button("Submit API Keys"):
        st.session_state["api_key"] = api_key
        st.session_state["env"] = env
        st.session_state["openai_api_key"] = openai_api_key
        st.session_state["cohere_api_key"] = cohere_api_key

# Main Application Flow
if all(
    key in st.session_state
    for key in ["api_key", "env", "openai_api_key", "cohere_api_key"]
):
    co, index = initialize_apis()
    if co and index:
        query = st.text_input("Enter search query:")
        top_k = st.number_input(
            "Top K resumes to fetch:", min_value=1, max_value=50, value=10
        )
        rerank_top_n = st.number_input(
            "Top N resumes to rerank:", min_value=1, max_value=top_k, value=5
        )

        if st.button("Search"):
            if query:
                with st.spinner("Fetching and evaluating resumes..."):
                    dataset = helpers.create_dataset()
                    helpers.insert_to_pinecone(index, dataset)
                    evaluation, error = helpers.evaluate_resumes(
                        index, co, query, top_k=top_k, rerank_top_n=rerank_top_n
                    )
                    comparison_data = helpers.compare(
                        index, co, query, top_k=top_k, top_n=rerank_top_n
                    )

                if evaluation:
                    st.markdown("### Evaluation:")
                    st.markdown(evaluation)
                    st.markdown("### Original vs Reranked Docs Comparison:")
                    st.write("---")
                    df_comparison = pd.DataFrame(comparison_data)
                    st.table(df_comparison)
                elif error:
                    st.warning(error)
            else:
                st.warning("Please enter a query.")
