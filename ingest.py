from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, AnswerParser, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores import WeaviateDocumentStore
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack import Pipeline

print("Import Successfully")

docs_path = ["data/atc manual - split.pdf"]

doc_store = WeaviateDocumentStore(host='http://localhost',
                                   port=8080,
                                   embedding_dim=768)

print("Document Store: ", doc_store)
print("#####################")

pdf_converter = PyPDFToDocument()
print("PDF Converter: ", pdf_converter)
print("#####################")
conversion_output = pdf_converter.run(paths=docs_path)
documents = conversion_output["documents"]
print("Documents: ", documents)
print("#####################")

processed_docs = []
for document in documents:
    print(document.text)
    new_document = {
        'content': document.text,
        'meta': document.metadata
    }
    processed_docs.append(new_document)
    print("#####################")

doc_preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
print("Document Preprocessor: ", doc_preprocessor)
print("#####################")

preprocessed_documents = doc_preprocessor.process(processed_docs)
print("Preprocessed Documents: ", preprocessed_documents)
print("#####################")

doc_store.write_documents(preprocessed_documents)


doc_retriever = EmbeddingRetriever(
    document_store=doc_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Document Retriever: ", doc_retriever)

doc_store.update_embeddings(doc_retriever)

print("Embeddings Done.")
