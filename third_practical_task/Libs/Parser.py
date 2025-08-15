from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.schema import Document


def parse_pdf(pdf_path: str | Path, include_page_breaks: bool = False, mode: str = 'single') -> list[Document] | None:
    pdf_path = Path(pdf_path)

    try:
        loader = UnstructuredPDFLoader(
            pdf_path.as_posix(),
            mode=mode,  # чтобы возвращалось как один объект
            strategy='auto',  # чтобы и сканы и обычные пдф читать
            languages=['rus', 'eng'],
            include_page_breaks=include_page_breaks  # склейка страниц
        )

        docs = loader.load()

        for d in docs:  # добавляем в метаданные имя файла
            d.metadata["source"] = pdf_path.name
            d.metadata["source_path"] = str(pdf_path)

        return docs

    except Exception as e:

        print(f'{pdf_path} ParseError: {e}')

        return None


def parse_pdfs_in_dir(
        dir_path: str | Path,
        recursive: bool = True,
        **kwargs,
) -> List[Document] | None:
    dir_path = Path(dir_path)

    pattern = "**/*.pdf" if recursive else "*.pdf"

    out: List[Document] = []

    for pdf in dir_path.glob(pattern):
        docs = parse_pdf(pdf, **kwargs)
        if docs:
            out.extend(docs)

    return out if out else print(f"Incorrect Path: {dir_path}")