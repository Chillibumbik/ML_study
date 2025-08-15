from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def make_chunks_from_document(doc: Document) -> list[Document] | None:
    try:
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", "", "? ", "! "],
                                                  chunk_size=500,
                                                  chunk_overlap=50)

        chunks = splitter.split_documents([doc])

        return chunks

    except Exception as e:
        print(f"Couldn't extract the chucks: {e}")
        return []
