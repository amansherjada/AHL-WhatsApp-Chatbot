import os
import json
from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

app = Flask(__name__)

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("ahlchatbot-customer")

# Load Vector Store from Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=openai_api_key,
    max_tokens=100,
    temperature=0.3,
    top_p=0.8,
)

# Custom Prompt Template
custom_prompt_template = """
MY TEMPLATE WILL BE HERE

{context}  

{question}  
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Create RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Log incoming request
        data = request.json
        print(f"Incoming Request from Gallabox: {json.dumps(data, indent=2)}")

        # Extract user message from the correct path
        event_type = data.get("event", "")
        contact = data.get("contact", {})
        whatsapp = data.get("whatsapp", {})

        user_message = None

        if "text" in whatsapp and "body" in whatsapp["text"]:
            user_message = whatsapp["text"]["body"]
        else:
            return jsonify({"error": "No message content provided"}), 400
        
        user_phone = whatsapp.get("from", "")

        if not user_message:
            return jsonify({"error": "No message content provided"}), 400

        # Process user message
        result = qa.invoke({"query": user_message})
        answer = result.get("result", "Sorry, I couldn't find an answer.")

        # Log AI response
        print(f"Received message from {user_phone}: {user_message}")
        print(f"Bot response: {answer}")

        # Ensure JSON response matches Gallabox's format
        response_data = {
            "status": "success",
            "data": {
                "messages": [{"text": answer}]
            }
        }

        return Response(json.dumps(response_data), status=200, mimetype="application/json")

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)