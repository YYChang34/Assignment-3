import os
import re
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import (
    get_embeddings, DATA_FOLDER, DB_FOLDER, FILES,
    EMBEDDING_MODEL_1, EMBEDDING_MODEL_2
)


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_db_folder(base_folder: str, model_name: str, chunk_size: int) -> str:
    model_short = model_name.split("/")[-1]
    return os.path.join(base_folder, f"{model_short}_chunk{chunk_size}")


def build_vector_dbs(
    model_name: str = EMBEDDING_MODEL_2,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> str:
    embeddings = get_embeddings(model_name)
    db_root = get_db_folder(DB_FOLDER, model_name, chunk_size)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    all_files = FILES.copy()
    already_mapped = set(FILES.values())
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".pdf") and f not in already_mapped:
            key = f.split(".")[0].lower()
            if key not in all_files:
                all_files[key] = f
                print(f"Found new document: {f} (key='{key}')")

    for key, filename in all_files.items():
        persist_dir = os.path.join(db_root, key)
        file_path = os.path.join(DATA_FOLDER, filename)

        if os.path.exists(persist_dir):
            continue

        if not os.path.exists(file_path):
            continue

        print(f"Building Vector Index for {key} (model={model_name}, chunk_size={chunk_size})")

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        splits = splitter.split_documents(docs)
        Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        print(f"Built DB for {key}")

    return db_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG vector databases from PDF files.")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL_2)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--all-experiments", action="store_true")
    args = parser.parse_args()

    if args.all_experiments:
        configs = [
            (EMBEDDING_MODEL_1, 500),
            (EMBEDDING_MODEL_1, 2000),
            (EMBEDDING_MODEL_1, 4000),
            (EMBEDDING_MODEL_2, 500),
            (EMBEDDING_MODEL_2, 2000),
            (EMBEDDING_MODEL_2, 4000),
        ]
        for model, chunk in configs:
            print(f"Building: model={model.split('/')[-1]}, chunk_size={chunk}")
            build_vector_dbs(model_name=model, chunk_size=chunk)
    else:
        build_vector_dbs(
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
