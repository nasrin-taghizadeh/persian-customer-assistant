import json
import os.path
import re
import requests
import numpy as np
from typing import Optional, Union, Type
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool, BaseTool
from langchain_core.documents import Document
from langchain.pydantic_v1 import BaseModel, Field


class LookUpPolicyInput(BaseModel):
    """ query of user """
    query: str = Field(description="query of user to be searched in policy database.")


class LookupPolicy(BaseTool):
    """ Look up frequently asked questions and policy """
    name: str = "lookup_policy"
    description: str = "Look up policy table of Alibaba group."
    args_schema: Type[BaseModel] = LookUpPolicyInput
    return_direct: bool = False

    llm: str = None
    retriever: Chroma = None
    data_dir: str = None

    def __init__(self, local_llm: str, data_dir):
        super().__init__()
        self.llm = local_llm
        self.data_dir = data_dir
        self.initialize_retriever()

    def initialize_retriever(self):
        embedding_model = OllamaEmbeddings(model=self.llm, num_thread=8, temperature=0.0)
        if os.path.exists(self.data_dir):
            self.retriever = Chroma(persist_directory=self.data_dir,
                                    embedding_function=embedding_model)
        else:
            with open(os.path.join(self.data_dir, "alibaba.md"), encoding="utf-8") as f:
                faq_text = "\n".join(f.readlines())
            docs = [Document(txt) for txt in re.split(r"(?=\n##)", faq_text)]
            vectorstore = Chroma.from_documents(
                documents=docs,
                collection_name="rag-chroma",
                embedding=embedding_model,
                persist_directory=self.data_dir
            )
            self.retriever = vectorstore

    def _run(self, query: str) -> str:
        """Consult the company policies to check whether certain options are permitted.
        Use this before making any flight changes performing other 'write' events."""
        docs = self.retriever.similarity_search(query)
        return "\n\n".join([doc.page_content for doc in docs])
