import os
import json
import time
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
import firebase_admin
from firebase_admin import credentials, firestore
import logging
from google.cloud.firestore_v1 import FieldFilter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

app = Flask(__name__)

# Initialize Firebase Admin SDK for Firestore
try:
    cred = credentials.Certificate(firebase_creds_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    # Fallback to a simple in-memory store for development/testing
    db = None
    conversation_memory = {}

# Configure session expiry time (in hours)
SESSION_TTL_HOURS = 24

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings()

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("ahlchatbot-customer")

# Load Vector Store from Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

# Custom Prompt Template with conversation history
custom_prompt_template = """
# American Hairline WhatsApp Customer Support AI Assistant  

## Core Objective  
Provide clear, friendly, and professional customer support for non-surgical hair replacement while guiding customers to connect with the team for a call.  

## **How to Start the Conversation (First Five Messages)**  
1. **Greet & Ask for Name**  
   - "Hi! Welcome to American Hairline 😊 What's your name?"  
2. **Ask for City**  
   - "Nice to meet you, [Name]! May I know which city you're from?"  
3. **Ask for Contact Preference**  
   - "Just so we can assist you better, do you prefer chatting here, or would you like to speak with our team on a call?"  
4. **Understand Their Inquiry**  
   - "What would you like to know about? I'm happy to help!"  
5. **Encourage Connecting with the Team**  
   - "I can give you basic info here, but for detailed guidance, our team can help you directly. You can call or WhatsApp them at +91 9222666111 anytime!"  

## **General Chat Guidelines**  
- **Keep it simple and natural** – no robotic language.  
- **Use short and clear messages** – don't overwhelm the customer.  
- **Make the conversation feel human** – warm and friendly, not like a bot.  

## **Handling Common Questions**  

### **Price Inquiries**  
❌ **Never share exact prices**  
✅ **How to respond:**  
  - "Pricing depends on your specific needs. The best way to get details is by speaking with our team. You can WhatsApp or call them at +91 9222666111."  

### **Location Inquiries**  
✅ **How to respond (Keep it short & friendly)**  
  - **Mumbai**: "We're at Saffron Building, Linking Rd, Khar West, Mumbai. Want to visit? You can WhatsApp us at +91 9222666111."  
  - **Delhi**: "We're in Greater Kailash-1, New Delhi (by appointment only). WhatsApp us to book a slot!"  
  - **Bangalore**: "We're in Indiranagar, Bangalore (by appointment only). Message us on WhatsApp to check availability!"  
  - **Other cities:** "We currently have stores in Mumbai, Delhi, and Bangalore. But we'd love to help—WhatsApp our team at +91 9222666111!"  

### **Product Questions**  
✅ **How to respond:**  
  - "We offer non-surgical hair replacement using real hair, customized to look completely natural. Let me know if you'd like more details!"  

### **Encouraging a Call**  
- The goal is to **suggest** a call naturally, without misleading.  
- Example:  
  - "I can help with basic info, but for the best advice, it's good to speak with our team. You can call or WhatsApp them at +91 9222666111 anytime!"  

## **Things NOT to Do**  
🚫 No medical advice.  
🚫 No competitor comparisons.  
🚫 No sharing personal client info.  
🚫 No exact pricing details.  

## Previous Conversation:
{history}

## Context from Knowledge Base:
{context}  

## Current Question:
{input}  

Remember to keep the conversation flowing naturally and maintain context from previous messages. Be helpful, friendly, and guide users toward connecting with the American Hairline team when appropriate.
"""

# Create a prompt template with all required variables
prompt = PromptTemplate(
    template=custom_prompt_template, 
    input_variables=["context", "input", "history"]
)

# Define your document combination chain
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create the retrieval chain
qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain,
)

def get_session_id(user_phone):
    """Generate or retrieve session ID for a user"""
    if db:
        # Check for existing active session
        sessions_ref = db.collection('sessions')
        # Update Firestore queries to use the 'filter' keyword argument
        input = sessions_ref.where(filter=FieldFilter('user_phone', '==', user_phone)).where(filter=FieldFilter('active', '==', True)).limit(1)
        results = input.get()
        
        if not results:
            # Create new session
            session_id = str(uuid.uuid4())
            expiry_time = datetime.now() + timedelta(hours=SESSION_TTL_HOURS)
            
            session_data = {
                'session_id': session_id,
                'user_phone': user_phone,
                'created_at': datetime.now(),
                'expires_at': expiry_time,
                'active': True,
                'metadata': {}
            }
            
            db.collection('sessions').document(session_id).set(session_data)
            return session_id
        else:
            # Return existing session
            session = results[0]
            return session.id
    else:
        # Fallback to in-memory storage
        if user_phone not in conversation_memory:
            session_id = str(uuid.uuid4())
            conversation_memory[user_phone] = {
                'session_id': session_id,
                'created_at': time.time(),
                'expires_at': time.time() + (SESSION_TTL_HOURS * 3600),
                'messages': [],
                'metadata': {}
            }
        return conversation_memory[user_phone]['session_id']

def get_conversation_history(session_id):
    """Retrieve conversation history for a session"""
    history = []
    
    if db:
        messages_ref = db.collection('sessions').document(session_id).collection('messages').order_by('timestamp')
        messages = messages_ref.get()
        
        for message in messages:
            msg_data = message.to_dict()
            if msg_data.get('role') == 'user':
                history.append(f"User: {msg_data.get('content')}")
            else:
                history.append(f"Assistant: {msg_data.get('content')}")
    else:
        # Fallback to in-memory storage
        for user_phone, data in conversation_memory.items():
            if data['session_id'] == session_id:
                history = data['messages'].copy()
                break
    
    # Return the last 10 messages to avoid token limits
    return history[-10:] if history else []

def save_message(session_id, role, content, user_phone=None):
    """Save a message to the conversation history"""
    timestamp = datetime.now()
    
    if db:
        message_data = {
            'role': role,
            'content': content,
            'timestamp': timestamp
        }
        
        # Save message to session
        db.collection('sessions').document(session_id).collection('messages').add(message_data)
        
        # Update session's last activity time
        session_ref = db.collection('sessions').document(session_id)
        session_ref.update({
            'last_activity': timestamp,
            'expires_at': timestamp + timedelta(hours=SESSION_TTL_HOURS)
        })
    else:
        # Fallback to in-memory storage
        if user_phone in conversation_memory:
            message = f"{role.capitalize()}: {content}"
            conversation_memory[user_phone]['messages'].append(message)
            conversation_memory[user_phone]['expires_at'] = time.time() + (SESSION_TTL_HOURS * 3600)

def cleanup_expired_sessions():
    """Clean up expired sessions from the database"""
    if db:
        current_time = datetime.now()
        expired_sessions = db.collection('sessions').where('expires_at', '<', current_time).where('active', '==', True).get()
        
        for session in expired_sessions:
            db.collection('sessions').document(session.id).update({
                'active': False
            })
    else:
        # Fallback to in-memory storage
        current_time = time.time()
        expired_keys = []
        
        for user_phone, data in conversation_memory.items():
            if data['expires_at'] < current_time:
                expired_keys.append(user_phone)
        
        for key in expired_keys:
            del conversation_memory[key]

def update_user_metadata(session_id, metadata):
    """Update user metadata in the session"""
    if db:
        session_ref = db.collection('sessions').document(session_id)
        current_metadata = session_ref.get().to_dict().get('metadata', {})
        updated_metadata = {**current_metadata, **metadata}
        
        session_ref.update({
            'metadata': updated_metadata
        })
    else:
        # Fallback to in-memory storage
        for user_phone, data in conversation_memory.items():
            if data['session_id'] == session_id:
                if 'metadata' not in data:
                    data['metadata'] = {}
                data['metadata'].update(metadata)
                break

def extract_user_info(message):
    """Extract potential user information from messages"""
    metadata = {}
    
    # This is a simple implementation - in production you might want to use
    # more sophisticated NLP to extract user details
    message_lower = message.lower()
    
    # Extract name
    name_indicators = ["my name is", "i am ", "i'm "]
    for indicator in name_indicators:
        if indicator in message_lower:
            pos = message_lower.find(indicator) + len(indicator)
            end_pos = message_lower.find(" ", pos + 5)  # Look for space after potential name
            if end_pos == -1:  # No space found, take the rest of the string
                end_pos = len(message_lower)
            name = message[pos:end_pos].strip()
            if len(name) > 2 and len(name) < 30:  # Basic validation
                metadata['name'] = name
                break
    
    # Extract city
    city_indicators = ["from ", "live in ", "located in ", "staying in "]
    cities = ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata", "pune", "ahmedabad"]
    for city in cities:
        if city in message_lower:
            metadata['city'] = city.capitalize()
            break
    
    return metadata

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Run session cleanup periodically
        if request.headers.get('X-Scheduled-Task') == 'cleanup':
            cleanup_expired_sessions()
            return jsonify({"status": "success", "message": "Cleanup completed"}), 200
        
        # Process incoming message
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

        # Get or create session
        session_id = get_session_id(user_phone)
        
        # Extract user information and update metadata
        metadata = extract_user_info(user_message)
        if metadata:
            update_user_metadata(session_id, metadata)
        
        # Get conversation history
        history = get_conversation_history(session_id)
        history_text = "\n".join(history) if history else "No previous conversation."
        
        # Save user message
        save_message(session_id, "user", user_message, user_phone)
        # Ensure the dictionary passed to qa.invoke() contains the required keys
        try:
            # Debugging: Print inputs
            print(f"User Message: {user_message}")
            print(f"History Text: {history_text}")

            # Process user message with conversation history
            result = qa_chain.invoke({
                "input": user_message, 
                "history": history_text,
                "context": retriever.invoke(user_message)
            })
            answer = result.get("result", "Sorry, I couldn't find an answer.")

        except Exception as e:
            print(f"Error in QA processing: {str(e)}")
            # Fallback response
            answer = "I'm having trouble processing your request right now. How can I help you with American Hairline's non-surgical hair replacement services?"
        
        # Save bot response
        save_message(session_id, "assistant", answer, user_phone)
        
        # Log the interaction
        print(f"Session {session_id} - Received from {user_phone}: {user_message}")
        print(f"Session {session_id} - Response: {answer}")

        # Return response in Gallabox's format
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the service"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)