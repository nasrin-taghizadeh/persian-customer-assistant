"""search in AliBaba policies"""
import os.path
import re
from typing import Type
from langchain_community.vectorstores import Chroma
from langchain_core.tools import BaseTool
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

    retriever: Chroma = None

    def __init__(self, data_dir: str, embedding_model):
        super().__init__()
        self.initialize_retriever(data_dir, embedding_model)

    def initialize_retriever(self, data_dir, embedding_model):
        if os.path.exists(data_dir):
            self.retriever = Chroma(persist_directory=data_dir,
                                    embedding_function=embedding_model)
        else:
            with open(os.path.join(data_dir, "alibaba.md"), encoding="utf-8") as f:
                faq_text = "\n".join(f.readlines())
            docs = [Document(txt) for txt in re.split(r"(?=\n##)", faq_text)]
            vectorstore = Chroma.from_documents(
                documents=docs,
                collection_name="rag-chroma",
                embedding=embedding_model,
                persist_directory=data_dir
            )
            self.retriever = vectorstore

    def _run(self, query: str) -> str:
        """Consult the company policies to check whether certain options are permitted.
        Use this before making any flight changes performing other 'write' events."""
        docs = self.retriever.similarity_search(query)
        return "\n\n".join([doc.page_content for doc in docs])

    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()


def get_tool_group_desc():
    return __doc__
