from flask import Flask, render_template, request
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)


bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")


bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

def get_documents():
    loader = PyPDFDirectoryLoader("pdf-data")  # all files in pdf -data input source
    documents = loader.load()

    
    for doc in documents:
        doc.metadata["source"] = doc.metadata.get("name", "//pdf-data//jiopayFAQ.pdf")

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)

    return docs


def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


docs = get_documents()
get_vector_store(docs)


faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()


llm = Bedrock(model_id="mistral.mistral-large-2402-v1:0", client=bedrock, model_kwargs={"temperature": 0.1})

prompt_template = """
Human: "You are a Jio Pay AI Support Assistant, designed to help users with their payment-related queries. 
Your goal is to provide clear, friendly, and helpful responses regarding Jio Pay services, transactions, 
account issues, and security. 

Please ensure:
- Responses are **short, professional, and accurate**.
- If the user asks about a **specific transaction**, guide them to check their **Jio Pay app**.
- If you **don't know an answer**, politely say so rather than guessing.

---

User Query: {question}

Relevant Information:
<context>
{context}
</context>

Jio Pay Assistant:
"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT}
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip().lower()  # Normalize input

 
    greetings = {"hi", "hello", "hey", "how are you", "good morning", "good evening"}
    if msg in greetings:
        return "Hello! How can I assist you with Jio Pay today? 😊"

    response = qa_chain({"query": msg})

    answer = response.get("result", "I'm sorry, but I couldn't find an answer. Please visit the Jio Pay help center.")
    sources = response.get("source_documents", [])

    
    if sources and msg not in greetings:
        citations = [doc.metadata.get("source", "//pdf-data//jiopayFAQ.pdf") for doc in sources]
        citations_text = "\n\nSources: " + ", ".join(set(citations))  # Avoid duplicates
        return answer + citations_text

    return answer  # No sources for greetings

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
