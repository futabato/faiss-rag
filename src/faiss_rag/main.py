from pathlib import Path

from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # or OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

base_dir = Path.cwd()
load_dotenv()

# Step 1: 文書読み込み
loader = PyMuPDFLoader(base_dir / "static" / "example_faq-manual-mPOS.pdf")
docs = loader.load()

# Step 2: テキストのチャンク分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 3: ベクトル化（Embedding）
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# embedding_model = OpenAIEmbeddings()  # OpenAI APIを使いたい場合はこちら

# Step 4: ベクトル格納（FAISSインデックス作成）
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Step 5: LLM連携（RetrievalQAチェーン作成）
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),  # または ChatOpenAI() にしてもOK
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Step 6: 質問応答
query = "この仕様書の35ページに書かれている、エラーコード107番の「発生した場合の対処法」を教えてください。"
print("Quety:", query)
answer = qa.run(query)
print("Answer:", answer)
