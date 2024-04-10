from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, AnswerParser, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores import WeaviateDocumentStore
from haystack import Pipeline
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack.preview.dataclasses import Document
from model_add import LlamaCPPLayer

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY_OPENAI = os.getenv("OPENAI_API_KEY")

print("Imported Successful")

web_app = FastAPI()

# Templates Configuration
template_engine = Jinja2Templates(directory="templates")

web_app.mount("/static", StaticFiles(directory="static"), name="static")

def search_documents(user_query):
    doc_storage = WeaviateDocumentStore(
        host='http://localhost',
        port=8080,
        embedding_dim=384
    )

    query_template = PromptTemplate(prompt="""Given the provided Documents, answer the Query. Make your answer detailed and long\n
                                                Query: {user_query}\n
                                                Documents: {join(documents)}
                                                Answer: 
                                            """,
                                            output_parser=AnswerParser())
    print("Query Template: ", query_template)
    
    def setup_model():
        return PromptModel(
            model_name_or_path="model/mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            invocation_layer_class=LlamaCPPLayer,
            use_gpu=False,
            max_length=512
        )

    # Query and Retrieval Logic (Truncated for brevity)

    # Print relevant information and return updated answer
    print("Relevant Documents:", relevant_docs)

    return updated_answer, relevant_docs

@web_app.get("/")
async def home(request: Request):
    return template_engine.TemplateResponse("index.html", {"request": request})

@web_app.post("/find_answer")
async def find_answer(request: Request, question: str = Form(...)):
    print(question)
    answer, related_docs = search_documents(question)
    response_info = jsonable_encoder(json.dumps({"answer": answer, "related_docs": related_docs}))
    response = Response(response_info)
    return response

# Uncomment for direct execution
# if __name__ == "__main__":
#     uvicorn.run("web_app:app", host='0.0.0.0', port=8001, reload=True)
