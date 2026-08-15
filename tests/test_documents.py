from dataclasses import dataclass

from langchain_core.documents import Document

from rag_app.documents import (
    decode_text_bytes,
    normalize_document_text,
    read_uploaded_documents,
    split_documents,
)


@dataclass
class FakeUpload:
    name: str
    type: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload

    def seek(self, _position: int) -> None:
        return None


def test_decode_text_bytes_supports_gb18030() -> None:
    assert decode_text_bytes("西游记".encode("gb18030")) == "西游记"


def test_normalize_document_text_fixes_compatibility_radicals() -> None:
    assert normalize_document_text("⽩⾻精") == "白骨精"


def test_read_uploaded_txt_documents_keeps_metadata() -> None:
    upload = FakeUpload("sample.txt", "text/plain", "本地文档".encode("utf-8"))

    documents = read_uploaded_documents([upload])

    assert len(documents) == 1
    assert documents[0].page_content == "本地文档"
    assert documents[0].metadata == {
        "source": "sample.txt",
        "source_type": "txt",
        "document_id": "upload-1",
    }


def test_read_uploaded_documents_distinguishes_same_name_uploads() -> None:
    uploads = [
        FakeUpload("sample.txt", "text/plain", "第一份".encode("utf-8")),
        FakeUpload("sample.txt", "text/plain", "第二份".encode("utf-8")),
    ]

    documents = read_uploaded_documents(uploads)

    assert [doc.metadata["document_id"] for doc in documents] == ["upload-1", "upload-2"]


def test_read_uploaded_pdf_pages_share_document_id(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakePdfReader:
        def __init__(self, _file) -> None:
            self.pages = [FakePage("第一页"), FakePage("第二页")]

    monkeypatch.setattr("rag_app.documents.PdfReader", FakePdfReader)
    upload = FakeUpload("sample.pdf", "application/pdf", b"fake pdf")

    documents = read_uploaded_documents([upload])

    assert [doc.metadata["document_id"] for doc in documents] == ["upload-1", "upload-1"]
    assert [doc.metadata["page"] for doc in documents] == [1, 2]


def test_split_documents_adds_chunk_id_and_preserves_source() -> None:
    documents = [
        Document(
            page_content="第一段\n第二段",
            metadata={
                "source": "sample.txt",
                "source_type": "txt",
                "document_id": "upload-1",
            },
        )
    ]

    chunks = split_documents(documents)

    assert chunks
    assert chunks[0].metadata["source"] == "sample.txt"
    assert chunks[0].metadata["source_type"] == "txt"
    assert chunks[0].metadata["document_id"] == "upload-1"
    assert chunks[0].metadata["chunk_id"] == 1
