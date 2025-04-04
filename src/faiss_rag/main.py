from pathlib import Path

from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # or OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import pdfplumber
from langchain.schema import Document

base_dir = Path.cwd()
load_dotenv()

# Step 1: 文書読み込み
pdf_path = "static/example_faq-manual-mPOS.pdf"
documents = []
# Step 2: テキストのチャンク分割
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if row and row[0] and row[0].strip().isdigit():
                    error_code = row[0].strip()
                    content = (row[1] or "").strip()
                    solution = (row[2] or "").strip()
                    full_text = f"コード: {error_code}\nエラーメッセージ: {content}\n内容と対策: {solution}"
                    documents.append(
                        Document(page_content=full_text, metadata={"page": i + 1})
                    )
for i, doc in enumerate(documents):
    print(f"Document {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    print()

# Step 3: ベクトル化（Embedding）
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: ベクトル格納（FAISSインデックス作成）
vectorstore = FAISS.from_documents(documents, embedding_model)

custom_prompt = PromptTemplate.from_template("""
あなたはアプリの仕様書のエラーコード一覧表を解析するアシスタントです。

以下に示すcontextには、「コード」「エラーメッセージ」「内容と対策」が表形式で記載されています。
コードは各行に1つずつあり、その横に内容と対処法が対応しています。

ユーザーからコードに関する質問があった場合、contextの中から該当するコードの行を探して、
その内容と対処法を正確に答えてください。

もしコードがcontextの中に存在しない場合は、「この文書には記載されていません」と答えてください。

<context>
{context}
</context>

質問: {question}
""")

# Step 5: LLM連携（RetrievalQAチェーン作成）
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k": 5,  # 上位1件の結果を取得
            "filter": {"page": 55},
        }  # 特定のページをフィルタリング
    ),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
)

# Step 6: 質問応答
query = "この仕様書に書かれている、コード「301」の「内容と対策」を教えてください。"
print("Query:", query)
answer = qa.run(query)
print("Answer:", answer)
