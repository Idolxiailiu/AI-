import streamlit as st
from parser import extract_text_from_pdf
from chunker import chunk_text
from embedding import get_embeddings
from vector_store import VectorStore
from rag_pipeline import rag_query

st.title("专利文档智能问答系统")

uploaded_file = st.file_uploader("上传PDF文件")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.write("解析PDF...")

    text = extract_text_from_pdf("temp.pdf")

    chunks = chunk_text(text)

    embeddings = get_embeddings(chunks)

    dimension = len(embeddings[0])

    vector_store = VectorStore(dimension)

    vector_store.add_embeddings(embeddings, chunks)

    st.success("文档解析完成")

    question = st.text_input("请输入你的问题")

    if question:

        answer = rag_query(question, vector_store)

        st.write("回答：")

        st.write(answer)