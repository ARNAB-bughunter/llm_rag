from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
import fitz

# embeddings = OllamaEmbeddings(model="phi4:latest",base_url="http://140.245.5.20:11434", num_thread=20)
embeddings = OllamaEmbeddings(model="llama3.2:3b")


vector_store_for_coord = InMemoryVectorStore(embeddings)


def get_rag_vector_embedding(docs):
    vector_store_for_llm = InMemoryVectorStore(embeddings)
    _ = vector_store_for_llm.add_documents(documents=docs)
    return vector_store_for_llm

def get_coord_vector_embedding(file_path, page_num):
    vector_store_for_coord = InMemoryVectorStore(embeddings)
    doc = fitz.open(file_path) # open a document
    
    page = doc[page_num]  # get a specific page
    text_data = page.get_text("dict")

    text_list_with_coord = []
    text_list = []
    
    for block in text_data["blocks"]:
        if block["type"] == 0:  # text block
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        bbox = span["bbox"]  # bounding box
                        text_list.append(text)
                        text_list_with_coord.append([text,bbox])
    
    embeddings_list = vector_store_for_coord.add_texts(texts=text_list)

    embeddings_coord_map = { i:j for i,j in zip(embeddings_list, text_list_with_coord)}

    return embeddings_coord_map, vector_store_for_coord