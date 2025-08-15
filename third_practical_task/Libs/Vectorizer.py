from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True,
                   'batch_size': 128}
)


def vectorizer(chunks: list[Document]) -> list[list[float]]:
    chunks_text = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.embed_documents(chunks_text)

    return embeddings
