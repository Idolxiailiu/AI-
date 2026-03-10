from parser import extract_text_from_pdf
from chunker import chunk_text
from embedding import get_embeddings
from vector_store import VectorStore

pdf_path = "data/patent.pdf"

print("读取PDF...")
text = extract_text_from_pdf(pdf_path)

print("文本分块...")
chunks = chunk_text(text)

print("生成Embedding...")
embeddings = get_embeddings(chunks)

print("构建FAISS...")
dimension = len(embeddings[0])
vector_store = VectorStore(dimension)

vector_store.add_embeddings(embeddings, chunks)

print("数据库构建完成")