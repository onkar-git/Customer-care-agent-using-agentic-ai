import re
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
        doc.metadata["source"] = doc.metadata.get("name", "//pdf-data//jiopayFAQ.pdf") # source

    
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

#setup the promts
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

    # Handle greetings separately
    greetings = {"hi", "hello", "hey", "how are you", "good morning", "good evening"}
    if msg in greetings:
        return "Hello! How can I assist you with Jio Pay today? 😊"

    # Handle acknowledgments and polite responses
    polite_responses = {"okay", "thanks", "thank you", "got it", "great", "cool"}
    if msg in polite_responses:
        return "You're welcome! Let me know if you need any more help. 😊"

    # Handle simple math questions like "2+2" or "5*3"
    if re.match(r"^\d+[\+\-\*/]\d+$", msg):
        try:
            result = eval(msg)  # Calculate safely
            return f"The answer is {result}."
        except:
            return "I'm sorry, I couldn't compute that."

    # Check if query is related to JioPay (Basic keyword check)
    jio_keywords = unique_words = {
    "1", "90", "able", "about", "account", "activate", "activities", "added", "aggregator", "all",
    "allow", "allowed", "allows", "almost", "alternative", "always", "am", "amount", "analytics", 
    "analyze", "android", "and", "answer", "any", "app", "app/dashboard", "appears", "application", 
    "apply", "approximately", "are", "as", "ask", "assistant", "at", "available", "back", "based", 
    "basic", "be", "because", "before", "benefit", "better", "between", "beyond", "both", "box", 
    "brings", "browser", "bulk", "business", "but", "by", "can", "cannot", "capabilities", "cards", 
    "case", "categories", "centralized", "certain", "check", "checking", "choose", "click", "cloud", 
    "collect", "collecting", "combining", "commercial", "common", "communication", "community", 
    "companies", "company", "compare", "complete", "completely", "compliance", "comprehensive", 
    "computer", "concept", "configure", "connection", "contact", "content", "continue", "control", 
    "convenience", "convenient", "corner", "create", "creating", "credentials", "crashing", "customer", 
    "customers", "customized", "dashboard", "data", "date", "default", "define", "delete", "delivering", 
    "dependencies", "depends", "describe", "design", "designed", "details", "determine", "device", 
    "devices", "different", "digital", "directly", "discuss", "display", "do", "does", "doesn", "download", 
    "downloading", "drive", "easily", "efficiency", "efficient", "either", "electronic", "email", "enable", 
    "enables", "encourage", "end", "ensure", "enters", "environment", "errors", "essential", "etc", "even", 
    "every", "evolution", "example", "excel", "execute", "experience", "exported", "express", "extensive", 
    "extra", "facing", "faq", "fast", "feature", "features", "feedback", "field", "fill", "final", "find", 
    "first", "focus", "follow", "following", "for", "forgot", "former", "forms", "free", "from", "fully", 
    "functionality", "future", "gain", "gateway", "general", "generate", "generated", "generating", "get", 
    "getting", "give", "go", "google", "government", "great", "growth", "guidance", "guide", "handle", 
    "happens", "has", "have", "having", "help", "helps", "here", "high", "highly", "history", "how", 
    "however", "https", "human", "icon", "idea", "identify", "if", "immediate", "impact", "implement", 
    "important", "improve", "in", "include", "includes", "including", "increasing", "indicated", "industry", 
    "information", "informed", "initiated", "insight", "insights", "install", "installation", "instructions", 
    "integrate", "integration", "intended", "interface", "internal", "internet", "into", "involved", "is", 
    "issues", "it", "item", "its", "jio", "jiopay", "job", "join", "keep", "key", "kit", "know", "known", 
    "language", "latest", "launch", "learn", "least", "let", "library", "like", "limited", "link", "list", 
    "live", "load", "local", "log", "login", "long", "look", "low", "mail", "major", "make", "making", 
    "management", "manage", "managing", "many", "market", "maximum", "may", "means", "meant", "medium", 
    "merchant", "merchants", "mentioned", "message", "migrate", "minimum", "mobile", "mode", "modes", 
    "modern", "monitor", "monitoring", "more", "most", "my", "name", "need", "needs", "net", "network", 
    "new", "next", "no", "not", "notified", "now", "number", "of", "offer", "offers", "offices", "offline", 
    "on", "once", "one", "online", "only", "open", "operate", "operation", "operations", "optimal", 
    "optimize", "or", "order", "organisation", "organize", "original", "other", "our", "out", "over", 
    "overview", "own", "page", "part", "partial", "password", "past", "pay", "payment", "payments", "pdf", 
    "people", "performance", "period", "phone", "phones", "platform", "play", "please", "plugin", "point", 
    "policy", "portal", "possible", "powerful", "practice", "preferred", "present", "prevent", "previous", 
    "primarily", "principle", "prior", "privacy", "probably", "process", "processes", "processing", "produce", 
    "products", "professional", "profile", "programming", "progress", "provide", "provided", "provider", 
    "providers", "provides", "providing", "public", "purchase", "purpose", "pwa", "question", "questions", 
    "quick", "range", "rate", "reach", "real", "reason", "receipts", "recognition", "recommend", "record", 
    "reduce", "refund", "refunds", "register", "regular", "reinstall", "related", "reliable", "reliance", 
    "remote", "remove", "reports", "required", "requirement", "requirements", "requires", "reset", "resource", 
    "respect", "response", "responsible", "restricted", "result", "retailers", "retrieve", "return", "review", 
    "right", "risk", "roadmap", "run", "safety", "same", "save", "scalable", "secure", "security", "see", 
    "seeks", "select", "self", "send", "sensitive", "separate", "service", "services", "session", "set", 
    "settlement", "several", "share", "shared", "short", "should", "show", "sign", "similar", "single", "site", 
    "size", "smartphone", "sms", "so", "social", "software", "solutions", "some", "something", "source", 
    "specific", "speed", "stability", "stable", "stage", "start", "state", "status", "steps", "still", "store", 
    "streamline", "strictly", "strong", "subject", "submission", "subscribe", "success", "such", "suitable", 
    "support", "switching", "system", "systems", "tap", "team", "technological", "technology", "tell", "text", 
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "thing", "this", "those", "through", 
    "time", "to", "today", "together", "tool", "top", "track", "trademark", "transaction", "transactions", 
    "trending", "trial", "tried", "troubleshooting", "true", "try", "two", "type", "ui", "unchecking", "under", 
    "understand", "undertake", "unique", "uninstall", "universal", "unknown", "unlike", "update", "updated", 
    "updates", "upon", "upside", "use", "used", "useful", "user", "users", "using", "value", "variety", 
    "various", "verification", "verify", "version", "very", "via", "video", "view", "views", "virtual", "visit", 
    "visual", "vital", "wallet", "way", "we", "web", "website", "well", "what", "when", "where", "which", 
    "while", "who", "why", "wifi", "will", "with", "within", "without", "work", "working", "world", "would", 
    "write", "writing", "wrong", "you", "your"}

    if not any(word in msg for word in jio_keywords):
        return "I'm sorry, but your query is not related to Jio Pay services. I can assist with jiopay payment-related questions. 😊"

    # Process retrieval-based queries
    response = qa_chain({"query": msg})

    answer = response.get("result", "I'm sorry, but I couldn't find an answer. Please visit the Jio Pay help center.")
    sources = response.get("source_documents", [])

    # Attach sources only if there are retrieved documents
    if sources:
        citations = [doc.metadata.get("source", "//pdf-data//jiopayFAQ.pdf") for doc in sources]
        citations_text = "\n\nSources: " + ", ".join(set(citations))
        return answer + citations_text

    return answer  # No sources for non-retrieved answers that means iy can be generilised



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
