from typing import Iterable, Optional
from pathlib import Path

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from langchain.schema import Document

from Libs import Chunker, Parser, Vectorizer

from dotenv import load_dotenv
import subprocess
import os


class DBInteracter:
    def __init__(
            self,
            client: Optional[OpenSearch] = None,
            host: str = "localhost",
            port: int = 9200,
            user: Optional[str] = None,
            password: Optional[str] = None,
            use_ssl: bool = False,
            index_name: str = "docs"
    ):
        self.index = index_name
        if client is not None:
            self.client = client
        else:
            self.client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(user, password) if user else None,
                use_ssl=use_ssl,
            )

    # ---------- index ----------

    def create_index(self, dim: int) -> None:
        if self.client.indices.exists(index=self.index):
            print("Index already exists")
            return

        body = {
            "settings": {
                "index.knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "parameters": {"ef_construction": 128, "m": 16}
                        }
                    }
                }
            }
        }
        self.client.indices.create(index=self.index, body=body)

    def drop_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)

    def refresh(self) -> None:
        self.client.indices.refresh(index=self.index)

    def count(self) -> int:
        return int(self.client.count(index=self.index).get("count", 0))

    # -------Dataprep -------

    @staticmethod
    def _bulk_actions(index: str, chunks: list[Document], embs: list[list[float]]) -> Iterable[dict[str, any]]:
        for ch, vec in zip(chunks, embs):
            content = getattr(ch, "page_content", str(ch))
            meta = getattr(ch, "metadata", {}) or {}
            action: dict[str, any] = {
                "_op_type": "index",
                "_index": index,
                "_source": {
                    "content": content,
                    "source": meta.get("source") or meta.get("file_path"),
                    "page": meta.get("page"),
                    "embedding": vec
                }
            }
            if "id" in meta:
                action["_id"] = meta["id"]
            yield action

    @staticmethod
    def _build_filters(filters: Optional[dict[str, any]]) -> list[dict[str, any]]:
        if not filters:
            return []
        out: list[dict[str, any]] = []
        for k, v in filters.items():
            if isinstance(v, dict):
                out.append({"range": {k: v}})
            else:
                out.append({"term": {k: v}})
        return out

    # ----- indexation -----

    def index_from_path(self, files_path: str | Path, batch_size: int = 500,
                        request_timeout: int = 120) -> tuple[int, list]:

        """
        Метод принимает на вход директорию, в которой содержатся файлы.
        Мы получаем чанки от файлов и их эмбеддингов,
        эти данные заносятся в индекс.
        На выходе мы получаем кортеж (успешно записано, ошибки)
        """
        documents = Parser.parse_pdfs_in_dir(files_path)

        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(Chunker.make_chunks_from_document(doc))

        if not chunks:
            print("No chunks parsed")
            return 0, []

        embs: list[list[float]] = Vectorizer.vectorizer(chunks)
        if not embs:
            raise RuntimeError("Vectorizer returned empty embeddings")
        dim = len(embs[0])

        self.create_index(dim=dim)

        # bulk индексирование
        success, errors = bulk(
            self.client,
            self._bulk_actions(self.index, chunks, embs),
            chunk_size=batch_size,
            request_timeout=request_timeout
        )
        self.refresh()
        return success, errors

    def index_chunks(self, chunks: list[Document], embs: list[list[float]],
                     batch_size: int = 500, request_timeout: int = 120) -> tuple[int, list]:

        """
        Получаем чанки от файлов и их эмбеддингов,
        эти данные заносятся в индекс.
        На выходе мы получаем кортеж (успешно записано, ошибки)
        """
        assert len(chunks) == len(embs), "len(chunks) != len(embs)"
        if not embs:
            raise RuntimeError("Empty embeddings")
        self.create_index(dim=len(embs[0]))
        success, errors = bulk(
            self.client,
            self._bulk_actions(self.index, chunks, embs),
            chunk_size=batch_size,
            request_timeout=request_timeout
        )
        self.refresh()
        return success, errors

    # ---------- Read/Delete -----------

    def get_doc(self, _id: str) -> Optional[dict[str, any]]:
        try:
            resp = self.client.get(index=self.index, id=_id)
            return resp.get("_source")
        except Exception:
            return None

    def delete_by_id(self, _id: str) -> dict[str, any]:
        return self.client.delete(index=self.index, id=_id, ignore=[404])

    def delete_by_source(self, source_value: str) -> dict[str, any]:
        body = {"query": {"term": {"source": source_value}}}
        return self.client.delete_by_query(index=self.index, body=body, refresh=True, conflicts="proceed")

    # ---------Search----------

    def search_text(
            self,
            query_text: str,
            size: int = 5,
            filters: Optional[dict[str, any]] = None,
            fields: Optional[list[str]] = None,
            highlight: bool = False
    ) -> dict[str, any]:

        """
        Принимаем запрос и количество векторов после поиска по тексту.
        На выходе получаем вектора
        """

        body: dict[str, any] = {
            "size": size,
            "query": {
                "bool": {
                    "must": [{"match": {"content": query_text}}],
                    "filter": self._build_filters(filters)
                }
            }
        }
        if fields:
            body["_source"] = fields
        if highlight:
            body["highlight"] = {"fields": {"content": {}}}
        return self.client.search(index=self.index, body=body)

    def search_knn(
            self,
            query_vec: list[float],
            k: int = 5,
            filters: Optional[dict[str, any]] = None,
            fields: Optional[list[str]] = None
    ) -> dict[str, any]:
        body: dict[str, any] = {
            "size": k,
            "query": {
                "bool": {
                    "must": [{"knn": {"embedding": {"vector": query_vec, "k": k}}}],
                    "filter": self._build_filters(filters)
                }
            }
        }
        if fields:
            body["_source"] = fields
        return self.client.search(index=self.index, body=body)

    def search_hybrid(
            self,
            query_text: str,
            query_vec: list[float],
            size: int = 5,
            knn_k: int = 5,
            filters: Optional[dict[str, any]] = None,
            fields: Optional[list[str]] = None
    ) -> dict[str, any]:
        body: dict[str, any] = {
            "size": size,
            "query": {
                "bool": {
                    "must": [{"knn": {"embedding": {"vector": query_vec, "k": knn_k}}}],
                    "should": [{"match": {"content": query_text}}],
                    "filter": self._build_filters(filters)
                }
            }
        }
        if fields:
            body["_source"] = fields
        return self.client.search(index=self.index, body=body)

    # ----------Query vectorization----------

    @staticmethod
    def vectorize_query(text: str) -> list[float]:
        return Vectorizer.vectorizer([text])[0]

    # -------DBLoader-----------

    @staticmethod
    def db_up(yml_path="DataBase.yml", env_path="Config.env") -> None:
        load_dotenv(env_path)

        try:
            print("Starting Docker Compose...")
            result = subprocess.run(["docker", "compose", "-f",
                                     yml_path, "--env-file", "Config.env", "up", "-d"], check=True,
                                    capture_output=True, text=True)

            print(result.stdout)

            return

        except Exception as e:
            print(e)
