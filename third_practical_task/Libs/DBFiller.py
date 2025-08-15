from opensearchpy import OpenSearch
from pathlib import Path
from langchain.schema import Document

from typing import Iterable

from opensearchpy.helpers import bulk

from Libs import Chunker
from Libs import Parser
from Libs import Vectorizer

def make_embeddings_from_path(files_path: str | Path) -> tuple[list, list]:
    """
    Функция принимает на вход директорию, в которой содержатся файлы.
    На выходе мы получаем кортеж из чанков от файлов и их эмбеддингов
    """

    documents = Parser.parse_pdfs_in_dir(files_path)

    chunks: list[Document] = []

    for doc in documents:
        chunks.extend(Chunker.make_chunks_from_document(doc))

    embeddings = Vectorizer.vectorizer(chunks)

    return chunks, embeddings


def create_index(client: OpenSearch, index: str, dim: int) -> None:
    """
    Создаем индекс, если его нет
    """

    if client.indices.exists(index=index):
        print('Index already exist')
        return

    body = {
        "settings": {
            "index.knn": True,          # включаем k-NN
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},         # сам текст чанка
                "source":  {"type": "keyword"},      # путь/метаданные
                "page":    {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {                       # базовая конфигурация
                        "name": "hnsw",
                        "engine": "nmslib",
                        "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 128, "m": 16}
                    }
                }
            }
        }
    }
    client.indices.create(index=index, body=body)


def bulk_prep(index: str, chunks: list, embs: list[list[float]]) -> Iterable[dict[str, any]]:
    """
    Подготовка данных для bulk
    """
    for ch, vec in zip(chunks, embs):
        content = getattr(ch, "page_content", str(ch))
        meta = getattr(ch, "metadata", {}) or {}
        yield {
            "_op_type": "index",
            "_index": index,
            "_source": {
                "content": content,
                "source": meta.get("source") or meta.get("file_path"),
                "page": meta.get("page"),
                "embedding": vec
            }
        }


def bulk_index(client: OpenSearch, index: str, chunks: list,
               embs: list[list[float]], batch_size: int = 500) -> tuple[int, list]:
    """
    Индексируем батчами. Возвращает (успешно_записано, ошибки).
    """
    assert len(chunks) == len(embs), "Число Эмбеддингов должно совпадать с числом Чанков"
    actions = list(bulk_prep(index, chunks, embs))
    success, errors = bulk(client, actions, chunk_size=batch_size, request_timeout=120)
    return success, errors


def make_index_and_fill(client: OpenSearch, files_path: str | Path, index_name: str) -> None:
    """
    Функция создает индекс и заполняет его данными из нужной нам директории
    """
    chunks, embeddings = make_embeddings_from_path(files_path)

    if not embeddings:
        raise RuntimeError("Эмбеддинги не получены (проверьте Vectorizer.vectorizer).")

    dim=len(embeddings[0])

    create_index(client, index_name, dim=dim)

    success, errors = bulk_index(client, index_name, chunks, embeddings)

    print(f"Indexed: {success}, errors: {errors}")
