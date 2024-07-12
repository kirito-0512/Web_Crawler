import streamlit as st
from pymilvus import connections, Collection
from data_retrieval import hybrid_retrieval, re_rank
from llm_interface import generate_answer

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("nvidia_docs")
collection.load()

st.title("NVIDIA CUDA Documentation QA System")

query = st.text_input("Enter your question:")

if query:
    retrieved_contexts = hybrid_retrieval(query, collection)
    
    if retrieved_contexts:
        re_ranked_contexts = re_rank(query, retrieved_contexts)
        answer = generate_answer(query, re_ranked_contexts)
        
        st.write("Answer:", answer)
        
        st.subheader("Top Retrieved and Re-ranked Documents:")
        for i, (url, content, score) in enumerate(re_ranked_contexts[:3]):
            st.write(f"{i+1}. Score: {score:.4f}")
            st.write(f"   URL: {url}")
            st.write(f"   Preview: {content[:150]}...")
            st.write("---")
    else:
        st.write("Sorry, I couldn't retrieve any relevant content to answer your question. Please try rephrasing your question or ask something else.")
