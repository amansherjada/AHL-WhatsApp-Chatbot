import os
import json
import time
import logging
import tempfile
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import secretmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Function to fetch secrets from Google Secret Manager
def access_secret(secret_name):
    """Fetch secret value from Google Secret Manager"""
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = "ahl-whatsapp-code"
        secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(name=secret_path)
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        raise e

# Check for required secrets
required_secrets = ["OPENAI_API_KEY", "PINECONE_API_KEY", "FIREBASE_CREDENTIAL_PATH"]
missing_secrets = [var for var in required_secrets if not access_secret(var)]
if missing_secrets:
    logger.error(f"Missing required secrets: {', '.join(missing_secrets)}")
    raise EnvironmentError(f"Missing required secrets: {', '.join(missing_secrets)}")

# Load API keys from Secret Manager
openai_api_key = access_secret("OPENAI_API_KEY")
pinecone_api_key = access_secret("PINECONE_API_KEY")
firebase_credential_json = access_secret("FIREBASE_CREDENTIAL_PATH")

# Write Firebase credentials to a temporary file
try:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(firebase_credential_json)
        firebase_credential_path = temp_file.name  # Get the file path
        logger.info(f"Temporary Firebase credential file created at {firebase_credential_path}")
except Exception as e:
    logger.error(f"Error writing Firebase credentials to file: {str(e)}")
    raise e

# Initialize Firebase
try:
    cred = credentials.Certificate(firebase_credential_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("🔥 Firebase initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing Firebase: {str(e)}")
    raise e

# Other configurations
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "ahlchatbot-customer")
llm_model = os.getenv("LLM_MODEL", "gpt-4o")
session_timeout_hours = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
retriever_k = int(os.getenv("RETRIEVER_K", "3"))

app = Flask(__name__)

# Initialize Firebase
try:
    cred = credentials.Certificate(firebase_credential_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    raise e

# Initialize Hugging Face embeddings
try:
    embedding_model = HuggingFaceEmbeddings()
    logger.info("HuggingFace embeddings initialized")
except Exception as e:
    logger.error(f"Error initializing embeddings: {str(e)}")
    raise e

# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    logger.info(f"Pinecone initialized with index: {pinecone_index_name}")
except Exception as e:
    logger.error(f"Error initializing Pinecone: {str(e)}")
    raise e

# Load Vector Store from Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})

# Initialize LLM
llm = ChatOpenAI(
    model=llm_model,
    openai_api_key=openai_api_key,
    max_tokens=200,
    temperature=0.3,
    top_p=0.8,
)
logger.info(f"LLM initialized with model: {llm_model}")

# Session management functions
def get_or_create_session(user_phone):
    """Retrieve existing session or create a new one"""
    try:
        session_ref = db.collection('sessions').document(user_phone)
        session = session_ref.get()
        
        if session.exists:
            # Update last activity timestamp
            session_ref.update({
                'last_activity': datetime.now().timestamp()
            })
            return session.to_dict()
        else:
            # Create new session
            new_session = {
                'user_phone': user_phone,
                'conversation_history': [],
                'created_at': datetime.now().timestamp(),
                'last_activity': datetime.now().timestamp()
            }
            session_ref.set(new_session)
            return new_session
    except Exception as e:
        logger.error(f"Error in get_or_create_session: {str(e)}")
        raise e

def update_conversation_history(user_phone, user_message, bot_response):
    """Add new messages to the conversation history"""
    try:
        session_ref = db.collection('sessions').document(user_phone)
        
        # Get current conversation history
        session = session_ref.get().to_dict()
        if not session:
            logger.error(f"Session not found for user: {user_phone}")
            return
            
        conv_history = session.get('conversation_history', [])
        
        # Add new messages
        timestamp = datetime.now().timestamp()
        conv_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': timestamp
        })
        conv_history.append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': timestamp
        })
        
        # Keep only the last N message pairs to manage context length
        if len(conv_history) > max_history_messages:
            conv_history = conv_history[-max_history_messages:]
        
        # Update the session
        session_ref.update({
            'conversation_history': conv_history,
            'last_activity': timestamp
        })
    except Exception as e:
        logger.error(f"Error in update_conversation_history: {str(e)}")
        raise e

def format_conversation_history(history):
    """Format conversation history for prompt"""
    if not history:
        return "No previous conversation."
    
    formatted = []
    for msg in history:
        role = "User" if msg['role'] == 'user' else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)

def cleanup_old_sessions():
    """Delete sessions that have been inactive for more than 24 hours"""
    try:
        cutoff_time = (datetime.now() - timedelta(hours=session_timeout_hours)).timestamp()
        old_sessions = db.collection('sessions').where('last_activity', '<', cutoff_time).stream()
        
        deleted_count = 0
        for session in old_sessions:
            session.reference.delete()
            deleted_count += 1
            
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} inactive sessions")
            
    except Exception as e:
        logger.error(f"Error cleaning up old sessions: {str(e)}")

def process_query_with_context(user_query, conversation_history):
    """Process a query with conversation context and retrieved documents"""
    try:
        # Get relevant documents from the retriever
        docs = retriever.get_relevant_documents(user_query)
        
        # Custom prompt template including conversation history
        prompt_template = """
        American Hairline WhatsApp Customer Support AI Assistant
        # You are a helpful WhatsApp assistant for AHL. Use the following retrieved context to answer the user's question. Be concise and professional.

        ## Core Objective
        Provide clear, friendly, and professional customer support for non-surgical hair replacement while guiding customers to connect with the team for a call.

        Keep your responses short and conversational, as if you were chatting with a customer on WhatsApp.

        ## General Chat Guidelines
          - Keep it simple and natural – no robotic language.
          - Use short and clear messages – don't overwhelm the customer.
          - Make the conversation feel human – warm and friendly, not like a bot.

        ## Handling Common Questions

        ### Price Inquiries
          ❌ Never share exact prices
          ✅ How to respond:
          - "Pricing depends on your specific needs. The best way to get details is by speaking with our team. You can WhatsApp or call them at +91 9222666111."

        ### Location Inquiries
          ✅ How to respond (Keep it short & friendly)
          - Mumbai: "We're at Saffron Building, 202, Linking Rd, opposite Satgurus Store, above Anushree Reddy Store, Khar, Khar West, Mumbai, Maharashtra 400052. Want to visit? You can WhatsApp us at +91 9222666111. Link = https://g.co/kgs/TJesmqE"
          - Delhi: "We're in Greater Kailash-1, New Delhi, but we see clients by appointment only. Please WhatsApp us to book a slot!"
          - Bangalore: "Our Indiranagar location in Bangalore operates by appointment only. Message us on WhatsApp to check availability and book!"
          - Other cities: "We currently have stores in Mumbai, Delhi, and Bangalore. But we'd love to help—WhatsApp our team at +91 9222666111!"

        ### Product Questions
          ✅ How to respond:
          - "We offer non-surgical hair replacement using real hair, customized to look completely natural. Let me know if you'd like more details!"

        ### Encouraging a Call
          - The goal is to suggest a call naturally, without misleading.
          - Example:
          - "I can share some general information here, but to discuss your specific needs and find the best solution, speaking with our team directly would be ideal. You can call or WhatsApp them at +91 9222666111."

        ## Things NOT to Do
          🚫 No medical advice.
          🚫 No competitor comparisons.
          🚫 No sharing personal client info.
          🚫 No exact pricing details.

        Retrieved context:
        {context}

        Conversation history:
        {conversation_history}

        User's current question: {question}

        Answer: Please let me know if you have any other basic questions before connecting with our team!
        """
        
        # Create the prompt
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "conversation_history", "question"]
        )
        
        # Create a chain that uses the custom prompt
        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
        
        # Run the chain
        result = qa_chain(
            {
                "input_documents": docs, 
                "conversation_history": conversation_history,
                "question": user_query
            }
        )
        
        return result["output_text"]
    except Exception as e:
        logger.error(f"Error in process_query_with_context: {str(e)}")
        # Return a fallback response in case of error
        return "I'm having trouble processing your request right now. Please try again in a moment or contact our team directly at +91 9222666111."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint for monitoring the health of the application"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get basic usage statistics"""
    try:
        # Count total active sessions
        active_sessions = db.collection('sessions').count().get()[0][0].value
        
        # Get total conversations in the last 24 hours
        day_ago = (datetime.now() - timedelta(hours=24)).timestamp()
        recent_sessions = db.collection('sessions').where('last_activity', '>', day_ago).stream()
        recent_count = sum(1 for _ in recent_sessions)
        
        return jsonify({
            "active_sessions": active_sessions,
            "sessions_last_24h": recent_count,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error generating stats: {str(e)}")
        return jsonify({"error": "Could not generate statistics"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Log incoming request
        data = request.json
        logger.info(f"Incoming Request from Gallabox")
        
        # Extract user message from the correct path
        whatsapp = data.get("whatsapp", {})

        user_message = None
        if "text" in whatsapp and "body" in whatsapp["text"]:
            user_message = whatsapp["text"]["body"]
        else:
            logger.error("No message content provided in request")
            return jsonify({"error": "No message content provided"}), 400
        
        user_phone = whatsapp.get("from", "")
        if not user_phone:
            logger.error("No user phone number provided in request")
            return jsonify({"error": "No user phone number provided"}), 400

        # Get or create session for this user
        session = get_or_create_session(user_phone)
        
        # Format conversation history for the prompt
        conversation_history = format_conversation_history(session.get('conversation_history', []))

        # Process user message with conversation history using our custom function
        answer = process_query_with_context(user_message, conversation_history)

        # Update conversation history
        update_conversation_history(user_phone, user_message, answer)

        # Log AI response
        logger.info(f"Processed message from {user_phone}")
        logger.debug(f"User message: {user_message}")
        logger.debug(f"Bot response: {answer}")

        # Ensure JSON response matches Gallabox's format
        response_data = {
            "status": "success",
            "data": {
                "messages": [{"text": answer}]
            }
        }

        # Trigger session cleanup occasionally
        if time.time() % 100 < 1:  # ~1% of requests
            cleanup_old_sessions()

        return Response(json.dumps(response_data), status=200, mimetype="application/json")

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        # Return a graceful error to the client
        return jsonify({
            "status": "error",
            "data": {
                "messages": [{
                    "text": "I'm having trouble processing your request right now. Please try again in a moment or contact our team directly at +91 9222666111."
                }]
            }
        }), 200  # Return 200 even on error so Gallabox can display the message

@app.route('/api/activity-log', methods=['GET'])
def activity_log():
    """Get recent activity logs"""
    try:
        # Get limit parameter with default of 10
        limit = request.args.get('limit', default=10, type=int)
        
        # Query the most recent sessions based on last_activity
        recent_sessions = db.collection('sessions')\
            .order_by('last_activity', direction=firestore.Query.DESCENDING)\
            .limit(limit)\
            .stream()
        
        activity_logs = []
        for session in recent_sessions:
            session_data = session.to_dict()
            # Get the last message exchange if any conversation history exists
            if session_data.get('conversation_history') and len(session_data['conversation_history']) >= 2:
                history = session_data['conversation_history']
                # Get the most recent user message and response pair
                last_user_msg = next((msg for msg in reversed(history) if msg['role'] == 'user'), None)
                
                if last_user_msg:
                    # Format phone number for privacy
                    phone = session_data.get('user_phone', '')
                    if len(phone) > 8:
                        formatted_phone = phone[:4] + "XXX" + phone[-3:]
                    else:
                        formatted_phone = "XXXXXXXX"
                    
                    # Calculate response time if we have both timestamps
                    response_time = "N/A"
                    for i, msg in enumerate(history):
                        if msg == last_user_msg and i+1 < len(history) and history[i+1]['role'] == 'assistant':
                            user_time = msg['timestamp']
                            bot_time = history[i+1]['timestamp']
                            response_time = f"{(bot_time - user_time):.1f}s"
                            break
                    
                    # Format the timestamp for display
                    timestamp = datetime.fromtimestamp(last_user_msg['timestamp'])
                    if datetime.now().date() == timestamp.date():
                        display_time = f"Today, {timestamp.strftime('%I:%M %p')}"
                    else:
                        display_time = timestamp.strftime('%b %d, %I:%M %p')
                    
                    activity_logs.append({
                        "time": display_time,
                        "phone": formatted_phone,
                        "status": "Completed",
                        "response_time": response_time
                    })
        
        return jsonify({
            "status": "success",
            "data": activity_logs,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error generating activity logs: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/cleanup', methods=['GET'])
def manual_cleanup():
    """Endpoint to manually trigger old session cleanup"""
    try:
        cleanup_old_sessions()
        return jsonify({"status": "success", "message": "Cleanup completed"}), 200
    except Exception as e:
        logger.error(f"Error in manual cleanup: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
