from pathlib import Path
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import pdfplumber

# Load environment variables
base_dir = Path.cwd()
load_dotenv()

# Constants
PDF_PATH = base_dir / "static/example_faq-manual-mPOS.pdf"
FAISS_INDEX_PATH = base_dir / "static/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Custom prompt template
CUSTOM_PROMPT = PromptTemplate.from_template("""
あなたはアプリの仕様書のエラーコード一覧表を解析するアシスタントです。

以下に示すcontextには、「エラーコード」「内容」「発生した場合の対処法」が表形式で記載されています。
エラーコードは各行に1つずつあり、その横に内容と対処法が対応しています。

以下の文書の中に該当があれば、厳密にその内容に従って答えてください
ユーザーからエラーコードに関する質問があった場合、contextの中から該当するエラーコードの行を探して、
以下の文書の中に該当があれば、厳密にその内容に従って答えてください。

contextの中に情報が存在しない場合は、「この文書には記載されていません」と答えてください。

<context>
{context}
</context>

質問: {question}
""")


def load_documents(pdf_path: Path) -> list[Document]:
    """
    Load documents from a PDF file and extract error codes, content, and solutions.
    :param pdf_path: Path to the PDF file
    :return: List of Document objects
    """
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            print(f"Page {i + 1} - tables: {tables}")
            for table in tables:
                for row in table:
                    if row and row[0] and re.match(r"^[A-Za-z0-9\-]+$", row[0].strip()):
                        error_code = row[0].strip()
                        content = (row[1] or "").strip()
                        solution = (row[2] or "").strip()
                        documents.append(
                            Document(
                                page_content=f"エラーコード: {error_code}\n内容: {content}\n対処法: {solution}",
                                metadata={
                                    "code": error_code,
                                    "domain": "mPOS",
                                    "page": i + 1,
                                },
                            )
                        )
    return documents


def create_vectorstore(
    documents: list[Document], model_name: str, index_path: Path
) -> FAISS:
    """
    Create and save a FAISS vectorstore from documents.
    :param documents: List of Document objects
    :param model_name: Name of the embedding model
    :param index_path: Path to save the FAISS index
    :return: FAISS vectorstore
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(index_path)
    return vectorstore


@lru_cache(maxsize=100)
def get_filtered_retriever_cached(vectorstore: FAISS, code: str) -> FAISS:
    """
    Retrieve a filtered retriever using caching.
    :param vectorstore: FAISS vectorstore
    :param code: Error code to filter by
    :return: Filtered retriever
    """
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "page": 35,
                "code": code,
            },
        }
    )


def get_answer(vectorstore: FAISS, error_code: str) -> str:
    """
    Generate an answer based on the error code.
    :param vectorstore: FAISS vectorstore
    :param error_code: Error code to query
    :return: Answer string
    """
    query = f"この仕様書に書かれている、エラーコード: {error_code}のときの「発生した場合の対処法」を教えてください。"
    retriever = get_filtered_retriever_cached(vectorstore, error_code)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("Query:", query)
        answer = "この文書には記載されていません。"
        print("Answer:", answer)
        return answer
    else:
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": CUSTOM_PROMPT},
        )
        print("Query:", query)
        answer = qa.run(query)
        print("Answer:", answer)
        return answer


if __name__ == "__main__":
    # Step 1: Load documents
    documents = load_documents(PDF_PATH)
    for i, doc in enumerate(documents):
        print(f"Document {i}:")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()

    # Step 2: Create vectorstore
    vectorstore = create_vectorstore(documents, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH)

    # Step 3: Query answers
    for code in ["108", "110", "3E5", "4F2"]:
        _ = get_answer(vectorstore, code)