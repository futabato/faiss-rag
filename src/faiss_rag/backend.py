from pathlib import Path
import re
import logging
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import pdfplumber


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
BASE_DIR = Path.cwd()
PDF_PATH = BASE_DIR.parent.parent / "static/example_faq-manual-mPOS.pdf"
FAISS_INDEX_PATH = BASE_DIR.parent.parent / "static/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Custom prompt template
CUSTOM_PROMPT = PromptTemplate.from_template("""
あなたはアプリの仕様書のエラーコード一覧表を解析するアシスタントです。

以下に示すcontextには、「エラーコード」「内容」「発生した場合の対処法」が表形式で記載されています。
エラーコードは各行に1つずつあり、その横に内容と対処法が対応しています。

以下の文書の中に該当があれば、厳密にその内容に従って答えてください。
contextの中に情報が存在しない場合は、「この文書には記載されていません」と答えてください。

<context>
{context}
</context>

質問: {question}
""")

app = FastAPI()


# --- Document Loading & Processing ---
def load_documents(pdf_path: Path) -> list[Document]:
    """
    PDFから文書を読み込み、エラーコード、内容、対処法を抽出してDocumentオブジェクトを生成する。
    """
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                print(f"Page {i + 1} - tables: {tables}")
                for table in tables:
                    for row in table:
                        if (
                            row
                            and row[0]
                            and re.match(r"^[A-Za-z0-9\-]+$", row[0].strip())
                        ):
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
    except Exception as e:
        logger.error(f"Error loading documents from {pdf_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDFの読み込みに失敗しました。",
        )
    return documents


def create_vectorstore(
    documents: list[Document], model_name: str, index_path: Path
) -> FAISS:
    """
    DocumentリストからFAISSインデックスを生成し、ローカルに保存する。
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(index_path)
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"FAISSインデックスの作成に失敗しました。",
        )
    return vectorstore


@lru_cache(maxsize=100)
def get_filtered_retriever_cached(vectorstore: FAISS, code: str) -> FAISS:
    """
    指定したエラーコードとページ番号でフィルタされたRetrieverをキャッシュして返す。
    """
    try:
        return vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {
                    "page": 35,
                    "code": code,
                },
            }
        )
    except Exception as e:
        logger.error(f"Error creating filtered retriever: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieverの作成に失敗しました。",
        )


def get_answer(vectorstore: FAISS, error_code: str) -> str:
    """
    エラーコードに基づいてRAGで回答を生成する。
    存在しない場合は定型の回答を返す。
    """
    query = f"この仕様書に書かれている、エラーコード: {error_code}のときの「発生した場合の対処法」を教えてください。"
    try:
        retriever = get_filtered_retriever_cached(vectorstore, error_code)
        docs = retriever.get_relevant_documents(query)
        if not docs:
            logger.info(f"Query: {query}")
            answer = "この文書には記載されていません。"
            logger.info(f"Answer: {answer}")
            return answer
        else:
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(temperature=0),
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": CUSTOM_PROMPT},
            )
            logger.info(f"Query: {query}")
            answer = qa.run(query)
            logger.info(f"Answer: {answer}")
            return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"回答の生成に失敗しました。",
        )


# --- FastAPI Startup: Load Documents and Create Vectorstore ---
@app.on_event("startup")
def startup_event():
    try:
        logger.info("Starting up: Loading documents from PDF...")
        documents = load_documents(PDF_PATH)
        logger.info(f"Loaded {len(documents)} documents.")
        logger.info("Creating vectorstore...")
        global vectorstore
        vectorstore = create_vectorstore(
            documents, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH
        )
        logger.info("Vectorstore created and ready.")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # 起動エラーの場合、FastAPIはクラッシュする。


# --- API Request & Response Models ---
class QueryRequest(BaseModel):
    error_code: str


class QueryResponse(BaseModel):
    error_code: str
    answer: str


# --- API Endpoint ---
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        answer = get_answer(vectorstore, request.error_code)
        return QueryResponse(error_code=request.error_code, answer=answer)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unhandled error in /query endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"サーバエラーが発生しました。",
        )
