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
    }


def test_split_documents_adds_chunk_id_and_preserves_source() -> None:
    documents = [
        Document(
            page_content="第一段\n第二段",
            metadata={"source": "sample.txt", "source_type": "txt"},
        )
    ]

    chunks = split_documents(documents)

    assert chunks
    assert chunks[0].metadata["source"] == "sample.txt"
    assert chunks[0].metadata["source_type"] == "txt"
    assert chunks[0].metadata["chunk_id"] == 1
