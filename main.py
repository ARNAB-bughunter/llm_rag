from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
import fitz


from vector import get_rag_vector_embedding, get_coord_vector_embedding
from graph_flow import graph_builder


def main(FILE_PATH, question):
    doc = fitz.open(FILE_PATH)
    graph = graph_builder()
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        doc = Document(page_content=text, metadata={"page": page_num})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents([doc])
        rag_vector_embedding = get_rag_vector_embedding(all_splits)
        embeddings_coord_map ,vector_store_for_coord = get_coord_vector_embedding(FILE_PATH, page_num)

        response = graph.invoke({"question": question,"rag_vector_embedding":rag_vector_embedding})
        retrieved_docs = vector_store_for_coord.similarity_search(response['answer'],k=4)

        for i in retrieved_docs:
            print(i,end="\n==============\n")

        print("answer",response['answer'])
        print(embeddings_coord_map.get(retrieved_docs[0].id))   


    


main("test.pdf","What is the total invoice amount in INR?")