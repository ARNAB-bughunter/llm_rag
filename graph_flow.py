from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

# llm = OllamaLLM(model="phi4:latest",base_url="http://140.245.5.20:11434", num_thread=20)
llm = OllamaLLM(model="llama3.2:3b")

template = (
        "You are tasked with extracting specific information from the following text context: {context}."
        "1. **Extract Information:** {question}"
        "2. **No Extra context:** Do not include any additional text, comments, or explanations in your response. "
        "3. **Empty Response:** If no information matches the description, return an empty string ('')."
        "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
        "5. **Max time 1 min:** You have max to max 1 min, to give response."
        "6. **Label not to be added:** do not add the label for the answer."
        "7. **Do Not Include Thought in the Answer:** do not send the the part of the answer where you talk about your thought <think> </think>."
    )



prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm 


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    rag_vector_embedding: InMemoryVectorStore

# Define application steps
def retrieve(state: State):
    retrieved_docs = state["rag_vector_embedding"].similarity_search(state["question"],k=2)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    
    return {"answer": response}


def graph_builder():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph