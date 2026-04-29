"""
RAG Pipeline - LangChain + HuggingFace Embeddings
Ingests ESG documents, builds a FAISS vector store,
and enables semantic Q&A over company filings.
"""

import os
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
import torch


class ESGRAGPipeline:
    """
    Core RAG pipeline for ESG document Q&A.
    Uses LangChain orchestration + HuggingFace sentence-transformers for embeddings.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[ESGRAGPipeline] Loading embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain = None
        print("[ESGRAGPipeline] Embeddings loaded")

    def ingest_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Ingest a list of {text, source, company} dicts into the vector store.
        Returns number of chunks indexed.
        """
        langchain_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                langchain_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": doc.get("source", "unknown"),
                        "company": doc.get("company", "unknown"),
                        "doc_type": doc.get("doc_type", "general"),
                        "chunk_id": i
                    }
                ))

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
        else:
            self.vectorstore.add_documents(langchain_docs)

        print(f"[ESGRAGPipeline] Indexed {len(langchain_docs)} chunks from {len(documents)} documents")
        return len(langchain_docs)

    def retrieve(self, query: str, k: int = 5, company_filter: Optional[str] = None) -> List[Document]:
        """Retrieve top-k relevant chunks for a query."""
        if self.vectorstore is None:
            raise ValueError("No documents indexed yet. Call ingest_documents() first.")

        docs = self.vectorstore.similarity_search(query, k=k * 2)

        if company_filter:
            docs = [d for d in docs if d.metadata.get("company", "").lower() == company_filter.lower()]

        return docs[:k]

    def query(self, question: str, company_filter: Optional[str] = None) -> Dict:
        """
        Answer a question using retrieved context.
        Returns answer + source documents for transparency.
        """
        relevant_docs = self.retrieve(question, k=5, company_filter=company_filter)

        if not relevant_docs:
            return {
                "question": question,
                "answer": "No relevant documents found for this query.",
                "sources": []
            }

        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'N/A')} | Company: {doc.metadata.get('company', 'N/A')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        # Simple extractive answer - highlights most relevant passage
        best_doc = relevant_docs[0]
        answer = (
            f"Based on available ESG disclosures:\n\n"
            f"{best_doc.page_content}\n\n"
            f"[Retrieved from: {best_doc.metadata.get('source', 'N/A')}]"
        )

        return {
            "question": question,
            "answer": answer,
            "context_used": context,
            "sources": [
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "company": doc.metadata.get("company", "N/A"),
                    "doc_type": doc.metadata.get("doc_type", "N/A"),
                    "excerpt": doc.page_content[:200] + "..."
                }
                for doc in relevant_docs
            ]
        }

    def save_index(self, path: str = "./esg_faiss_index"):
        """Persist the FAISS index to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"[ESGRAGPipeline] Index saved to {path}")

    def load_index(self, path: str = "./esg_faiss_index"):
        """Load a persisted FAISS index."""
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"[ESGRAGPipeline] Index loaded from {path}")
