import os
import json
import tempfile
import re
import pickle
import numpy as np
import random
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter

# Google Maps API
try:
    import googlemaps
    GOOGLE_MAPS_AVAILABLE = True
except ImportError:
    GOOGLE_MAPS_AVAILABLE = False
    logging.warning("Google Maps library not installed. Install with: pip install googlemaps")

# Suppress warnings
warnings.filterwarnings("ignore", message="Detected filter using positional arguments")

# ========== SETUP ==========
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins for now
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
# ========== ROOT ROUTE ==========
@app.route('/')
def home():
    """Root endpoint to confirm service is running"""
    return jsonify({
        "service": "Bah.AI Property Chatbot API",
        "status": "online",
        "version": "3.7.0",  # Updated version
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/": "Service status (GET)",
            "/api/chat": "Chatbot endpoint (POST)",
            "/api/health": "Health check (GET)",
            "/api/test": "Test model (GET)"
        },
        "features": {
            "general_searches": True,
            "criteria_searches": True,
            "financing_queries": True,
            "document_queries": True,
            "landmark_proximity": True,
            "description_checking": True
        }
    })
@app.route('/api/debug-files', methods=['GET'])
def debug_files():
    """Debug endpoint to check file paths"""
    import os
    
    debug_info = {
        'current_directory': os.getcwd(),
        'script_directory': BASE_DIR,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'training_data_path': TRAINING_DATA_PATH,
        'training_data_exists': os.path.exists(TRAINING_DATA_PATH),
    }
    
    # Check models directory
    models_dir = os.path.join(BASE_DIR, 'models')
    if os.path.exists(models_dir):
        debug_info['models_directory'] = os.listdir(models_dir)
    else:
        debug_info['models_directory'] = 'Directory not found'
    
    # Check data directory
    data_dir = os.path.join(BASE_DIR, 'data')
    if os.path.exists(data_dir):
        debug_info['data_directory'] = os.listdir(data_dir)
        
        # Check member1 subdirectory
        member1_dir = os.path.join(data_dir, 'member1')
        if os.path.exists(member1_dir):
            debug_info['member1_directory'] = os.listdir(member1_dir)
        else:
            debug_info['member1_directory'] = 'member1 directory not found'
    else:
        debug_info['data_directory'] = 'Directory not found'
    
    return jsonify(debug_info)
# ========== CONFIGURATION ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(os.path.dirname(BASE_DIR), "training")

# Use absolute paths
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'training', 'models', 'nlu_model.pkl')
TRAINING_DATA_PATH = os.path.join(TRAINING_DIR, 'data', 'member1', 'training_data.json')

print(f"\n📁 BASE_DIR: {BASE_DIR}")
print(f"📁 TRAINING_DIR: {TRAINING_DIR}")
print(f"📁 MODEL_PATH: {MODEL_PATH}")
print(f"📁 TRAINING_DATA_PATH: {TRAINING_DATA_PATH}")
print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")
print(f"📁 Training data exists: {os.path.exists(TRAINING_DATA_PATH)}")

# Debug: List files in directories
print("\n🔍 Checking directories...")
if os.path.exists(os.path.join(BASE_DIR, 'models')):
    print(f"📂 Files in models directory:")
    for f in os.listdir(os.path.join(BASE_DIR, 'models')):
        print(f"   - {f}")
else:
    print("❌ models directory not found!")

if os.path.exists(os.path.join(BASE_DIR, 'data')):
    print(f"📂 Files in data directory:")
    for f in os.listdir(os.path.join(BASE_DIR, 'data')):
        print(f"   - {f}")
else:
    print("❌ data directory not found!")

# Global variables
vectorizer = None
classifier = None
db = None
nlp = None
model_classes = []
training_data = {}

# ========== FIREBASE INITIALIZATION ==========
print("\n" + "="*60)
print("🔥 FIREBASE CONNECTION")
print("="*60)

def initialize_firebase():
    """Initialize Firebase using environment variable or file"""
    global db
    
    # First try: Environment variable method (for Render)
    firebase_key_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
    
    if firebase_key_json:
        print("✅ Found Firebase key in environment variable")
        try:
            # Parse JSON from environment variable
            key_data = json.loads(firebase_key_json)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(key_data, f)
                temp_key_path = f.name
            
            print(f"🔑 Using temporary key file from environment variable")
            
            # Check if Firebase is already initialized
            if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
                print("⚠️  Firebase already initialized, using existing app")
                db = firestore.client()
            else:
                # Initialize with the temporary file
                cred = credentials.Certificate(temp_key_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': key_data.get('project_id', 'bahai-1b76d'),
                    'databaseURL': 'https://bahai-1b76d.firebaseio.com',
                    'storageBucket': 'bahai-1b76d.appspot.com',
                })
                db = firestore.client()
            
            print(f"📋 Project ID: {key_data.get('project_id')}")
            print("✅ Firebase initialized successfully from environment variable")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in environment variable: {e}")
        except Exception as e:
            print(f"❌ Firebase initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    # Second try: Check for key file (for local development)
    print("🔍 Checking for serviceAccountKey.json file...")
    file_paths = [
        'serviceAccountKey.json',
        '/opt/render/project/src/serviceAccountKey.json',
        '../serviceAccountKey.json',
        './serviceAccountKey.json'
    ]
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"📁 Found key file at: {file_path}")
            try:
                # Check if Firebase is already initialized
                if firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:
                    print("⚠️  Firebase already initialized, using existing app")
                    db = firestore.client()
                else:
                    # Initialize with the file
                    cred = credentials.Certificate(file_path)
                    firebase_admin.initialize_app(cred)
                    db = firestore.client()
                
                print("✅ Firebase initialized successfully from file")
                return True
            except Exception as e:
                print(f"❌ Error loading Firebase key from {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    print("⚠️  WARNING: Firebase not initialized")
    print("💡 Make sure to set FIREBASE_SERVICE_ACCOUNT_KEY environment variable in Render")
    db = None
    return False

# Initialize Firebase
firebase_initialized = initialize_firebase()

# TEST Firebase connection
if firebase_initialized and db:
    try:
        print("🔍 Testing Firestore connection...")
        properties_ref = db.collection('properties')
        docs = list(properties_ref.limit(5).get())
        print(f"📊 Found {len(docs)} properties in database")
        
        if docs:
            print("✅ Firestore connection successful!")
        else:
            print("⚠️ No properties found in database (may be empty)")
    except Exception as e:
        print(f"⚠️ Firestore query warning: {e}")
        print("💡 Connection established but query failed")
else:
    print("❌ Firebase not connected - using mock data mode")

nlp = None


# ========== LOAD TRAINING DATA ==========
def load_training_data():
    """Load training data for response templates"""
    global training_data
    
    print(f"\n🔍 Attempting to load training data from: {TRAINING_DATA_PATH}")
    print(f"🔍 File exists: {os.path.exists(TRAINING_DATA_PATH)}")
    
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            logger.info(f"✅ Training data loaded from {TRAINING_DATA_PATH}")
            
            if 'location_profiles' in training_data:
                logger.info(f"📊 Found {len(training_data['location_profiles'])} location profiles")
        else:
            logger.warning(f"⚠️ Training data file not found: {TRAINING_DATA_PATH}")
            training_data = {}
    except Exception as e:
        logger.error(f"❌ Error loading training data: {e}")
        training_data = {}
# ========== LOAD NLU MODEL ==========
def load_nlu_model():
    """Load the trained NLU model from train_nlu.py"""
    global vectorizer, classifier, model_classes
    
    print(f"\n🔍 Attempting to load model from: {MODEL_PATH}")
    print(f"🔍 File exists: {os.path.exists(MODEL_PATH)}")
    
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"📂 Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            vectorizer = model_data.get('vectorizer')
            classifier = model_data.get('classifier')
            
            if classifier and hasattr(classifier, 'classes_'):
                model_classes = classifier.classes_.tolist()
                logger.info(f"✅ NLU model loaded successfully (v{model_data.get('version', '1.0')})")
                logger.info(f"📊 Model intents: {model_classes}")
                logger.info(f"📊 Feature count: {len(vectorizer.get_feature_names_out()) if vectorizer else 0}")
            else:
                logger.warning("⚠️ Classifier doesn't have classes_ attribute")
                
        else:
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            logger.error("💡 Run train_nlu.py first to create the model!")
            
    except Exception as e:
        logger.error(f"❌ Error loading NLU model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

# ========== TEXT PREPROCESSING ==========
def preprocess_text(text):
    """Preprocess text for prediction"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s\?\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ========== ENTITY EXTRACTION ==========
def classify_landmark_type(landmark: str) -> str:
    """Classify the type of landmark"""
    landmark_lower = landmark.lower()
    
    # School-related keywords
    school_keywords = ['school', 'university', 'college', 'campus', 'academy', 'institute', 'elementary', 
                      'high school', 'highschool', 'primary school', 'secondary school', 'univ', 'col', 
                      'education', 'student', 'academic']
    for keyword in school_keywords:
        if keyword in landmark_lower:
            return 'school'
    
    # Hospital-related keywords
    hospital_keywords = ['hospital', 'medical center', 'clinic', 'health center', 'healthcare', 'medical', 
                        'infirmary', 'doctor', 'health']
    for keyword in hospital_keywords:
        if keyword in landmark_lower:
            return 'hospital'
    
    # Mall/shopping keywords
    mall_keywords = ['mall', 'shopping', 'shopping center', 'commercial center', 'market', 'sm ', 'robinsons', 
                    'ayala mall', 'department store', 'supermarket']
    for keyword in mall_keywords:
        if keyword in landmark_lower:
            return 'mall'
    
    # Transportation keywords
    transport_keywords = ['port', 'pier', 'terminal', 'bus terminal', 'airport', 'train station', 'lrt', 
                         'mrt', 'transportation', 'bus station']
    for keyword in transport_keywords:
        if keyword in landmark_lower:
            return 'transportation'
    
    # Beach/nature keywords
    nature_keywords = ['beach', 'volcano', 'lake', 'mountain', 'park', 'garden', 'resort', 'taal', 'nature', 
                      'sea', 'ocean', 'river']
    for keyword in nature_keywords:
        if keyword in landmark_lower:
            return 'nature'
    
    # Church/religious keywords
    church_keywords = ['church', 'cathedral', 'basilica', 'chapel', 'temple', 'mosque', 'religious', 'worship']
    for keyword in church_keywords:
        if keyword in landmark_lower:
            return 'church'
    
    return 'general'

def extract_entities_from_query(query: str) -> Dict[str, Any]:
    """Extract entities from user query"""
    entities = {
        'property_type': None,
        'location': None,
        'landmark': None,
        'landmark_type': None,
        'feature': None,  
        'price_range': None, 
        'bedrooms': None,
        'bathrooms': None,
        'financing_type': None,
        'listing_type': None,
        'sale_type': None,
        'financing_options': None,
        'has_general_search': False,
        'max_price': None,
        'min_price': None,
        'min_bedrooms': None,
        'exact_bedrooms': None,
        'documents_only': False,
        'documents_info': None,
        'family_info': None,
        'has_need_query': False,
        'proximity': 'near',
        'google_maps_check': False,
    }
    
    query_lower = query.lower()

    # Detect family needs
    family_patterns = [
        (r'family\s+of\s+(\d+)', 'family_size'),
        (r'family\s+with\s+(\d+)', 'family_size'),
        (r'(\d+)\s+person\s+family', 'family_size'),
        (r'(\d+)-member\s+family', 'family_size'),
        (r'(\d+)\s+people\s+family', 'family_size'),
        (r'small\s+family', 'small_family'),
        (r'big\s+family', 'big_family'),
        (r'large\s+family', 'big_family'),
        (r'young\s+family', 'small_family'),
        (r'growing\s+family', 'medium_family'),
        (r'family\s+with\s+kids', 'family_with_kids'),
        (r'family\s+with\s+children', 'family_with_kids'),
        (r'\bcouple\b', 'couple'),
        (r'\bcouples\b', 'couple'),
        (r'\bfor\s+couple\b', 'couple'),
        (r'\bfor\s+couples\b', 'couple'),
        (r'\btwo\s+person\b', 'couple'),
        (r'\bhusband\s+and\s+wife\b', 'couple'),
    ]
    
    for pattern, family_type in family_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if family_type == 'family_size':
                family_size = int(match.group(1))
                entities['family_info'] = {'type': 'size', 'value': family_size}
                
                # Set minimum bedroom requirements based on family size
                if family_size <= 2:
                    entities['min_bedrooms'] = 1  # Couple or 1 child
                    entities['ideal_bedrooms'] = 2
                    logger.info(f"👨‍👩‍👧‍👦 Family of {family_size} → 1-2 bedrooms recommended")
                elif 3 <= family_size <= 4:
                    entities['min_bedrooms'] = 2  # Minimum for family of 3-4
                    entities['ideal_bedrooms'] = 3
                    logger.info(f"👨‍👩‍👧‍👦 Family of {family_size} → 2-3 bedrooms recommended")
                elif 5 <= family_size <= 6:
                    entities['min_bedrooms'] = 3  # Minimum for family of 5-6
                    entities['ideal_bedrooms'] = 4
                    logger.info(f"👨‍👩‍👧‍👦 Family of {family_size} → 3-4 bedrooms recommended")
                else:  # 7+ people
                    entities['min_bedrooms'] = 4  # Minimum for large family
                    entities['ideal_bedrooms'] = 5
                    logger.info(f"👨‍👩‍👧‍👦 Family of {family_size} → 4+ bedrooms recommended")
            else:
                entities['family_info'] = {'type': family_type}
                
                # For generic family types
                if family_type in ['big_family', 'large_family']:
                    entities['min_bedrooms'] = 4
                    entities['ideal_bedrooms'] = 5
                    logger.info(f"👨‍👩‍👧‍👦 Large family → 4+ bedrooms recommended")
                elif family_type in ['small_family', 'young_family', 'couple']: 
                    entities['min_bedrooms'] = 1
                    entities['ideal_bedrooms'] = 2
                    logger.info(f"👨‍👩‍👧‍👦 Small family/couple → 1-2 bedrooms recommended")
                elif family_type == 'medium_family':
                    entities['min_bedrooms'] = 3
                    entities['ideal_bedrooms'] = 4
                    logger.info(f"👨‍👩‍👧‍👦 Medium family → 3-4 bedrooms recommended")
                elif family_type == 'family_with_kids':
                    entities['min_bedrooms'] = 2
                    entities['ideal_bedrooms'] = 3
                    logger.info(f"👨‍👩‍👧‍👦 Family with kids → 2-3 bedrooms recommended")
                    
            logger.info(f"👨‍👩‍👧‍👦 Detected family need: {entities['family_info']}")
            break
    
    # Detect needs-based queries
    needs_keywords = ['for family', 'for students', 'for professionals', 'for couple', 
                    'for couples', 'for retirees', 'for business', 'for investors', 'for single', 
                    'for workers', 'for office', 'for commercial']
    
    has_needs_keyword = any(keyword in query_lower for keyword in needs_keywords)
    
    if has_needs_keyword:
        entities['has_need_query'] = True
        logger.info("🎯 Marked as needs-based query")
    
    # Detect document-only queries
    doc_keywords = ['documents', 'requirements', 'needed', 'required', 'paperwork', 'papers', 'what do i need']
    prop_keywords = ['properties', 'show me', 'find', 'looking for', 'search', 'houses', 'condos', 'apartments', 'property']
    
    has_doc_keywords = any(term in query_lower for term in doc_keywords)
    has_prop_keywords = any(term in query_lower for term in prop_keywords)
    
    if has_doc_keywords and not has_prop_keywords:
        entities['documents_only'] = True
        entities['documents_info'] = True
        logger.info("📋 Detected document-only query")
    
    if 'what documents' in query_lower or 'what are the requirements' in query_lower or 'what do i need' in query_lower:
        entities['documents_only'] = True
        entities['documents_info'] = True
        logger.info("📋 Detected 'what' document query")
    
    # Parse numeric price values
    max_price = None
    min_price = None
    
    price_patterns = [
        (r'under\s+(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'max'),
        (r'below\s+(\d+(?:\.\d+)?)\s*million\b', lambda m: float(m.group(1)) * 1000000, 'max'),
        (r'(?:under|below)\s*₱?\s*(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'max'),
        (r'(?:under|below)\s+(\d{7,})\b', lambda m: float(m.group(1)), 'max'),
        (r'less\s+than\s+(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'max'),
        (r'maximum\s+(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'max'),
        (r'up\s+to\s+(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'max'),
        (r'(?:above|over)\s+(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'min'),
        (r'minimum\s+(\d+(?:\.\d+)?)\s*([mM])\b', lambda m: float(m.group(1)) * 1000000, 'min'),
        (r'(?:from|between)\s+(\d+(?:\.\d+)?)\s*([mM])?\s*(?:to|and)\s+(\d+(?:\.\d+)?)\s*([mM]?)', 
         lambda m: (float(m.group(1)) * (1000000 if m.group(2) else 1), 
                   float(m.group(3)) * (1000000 if m.group(4) else 1)), 'range'),
        (r'\b(\d+(?:\.\d+)?)\s*([mM])\b(?!\s*(?:bed|bedroom|bath))', 
         lambda m: float(m.group(1)) * 1000000, 'exact'),
    ]
    
    for pattern, converter, price_type in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                if price_type == 'max':
                    max_price = converter(match)
                    entities['max_price'] = max_price
                    entities['price_range'] = f"under ₱{max_price/1000000:.1f}M"
                    logger.info(f"💰 Parsed max price: ₱{max_price:,.0f}")
                elif price_type == 'min':
                    min_price = converter(match)
                    entities['min_price'] = min_price
                    logger.info(f"💰 Parsed min price: ₱{min_price:,.0f}")
                elif price_type == 'range':
                    min_val, max_val = converter(match)
                    entities['min_price'] = min_val
                    entities['max_price'] = max_val
                    entities['price_range'] = f"₱{min_val/1000000:.1f}M to ₱{max_val/1000000:.1f}M"
                    logger.info(f"💰 Parsed price range: ₱{min_val:,.0f} - ₱{max_val:,.0f}")
                elif price_type == 'exact':
                    exact_price = converter(match)
                    entities['price_range'] = f"around ₱{exact_price/1000000:.1f}M"
                    logger.info(f"💰 Parsed approximate price: ₱{exact_price:,.0f}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Could not parse price pattern '{pattern}': {e}")
                continue
    
    # Parse bedroom criteria
    bedroom_patterns = [
        (r'with\s+(\d+)\s+beds?\b', lambda m: int(m.group(1))),  
        (r'with\s+(\d+)\s+bedroom(?:s)?\b', lambda m: int(m.group(1))),
        (r'\b(\d+)\s+beds?\b(?!\s*(?:bath|bathroom))', lambda m: int(m.group(1))), 
        (r'\b(\d+)\s+bedroom(?:s)?\b(?!\s*(?:bath|bathroom))', lambda m: int(m.group(1))),
        (r'(\d+)(?:-|\s*)bedroom|(\d+)br\b', lambda m: int(m.group(1)) if m.group(1) else int(m.group(2))),
        (r'(\d+)\s+bed\b', lambda m: int(m.group(1))),
        (r'\bstudio\b', lambda m: 0),
        (r'(\d+)\s+bedroom\s+(?:apartment|condo|house|unit)', lambda m: int(m.group(1))),
    ]
    
    for pattern, converter in bedroom_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                bedrooms = converter(match)
                entities['exact_bedrooms'] = bedrooms
                entities['bedrooms'] = bedrooms
                logger.info(f"🛏️ Parsed bedroom count: {bedrooms}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Could not parse bedroom pattern '{pattern}': {e}")
                continue
    
    # Parse bathroom criteria
    bathroom_patterns = [
        (r'(\d+)\s+baths?', 'bathrooms'),
        (r'(\d+)\s+bathrooms?', 'bathrooms'),
        (r'with\s+(\d+)\s+bath', 'bathrooms'),
        (r'(\d+)\s+ba', 'bathrooms')
    ]
    
    for pattern, entity_type in bathroom_patterns:
        match = re.search(pattern, query_lower)  
        if match:
            entities[entity_type] = int(match.group(1))  
            break
    
    # Detect location
    has_location_terms = any(term in query_lower for term in ['in ', 'at ', 'within ', 'inside '])
    has_specific_location = False
    
    # Detect listing type
    if 'for rent' in query_lower or 'rental' in query_lower:
        entities['listing_type'] = 'rent'
    elif 'for sale' in query_lower or 'buy' in query_lower:
        entities['listing_type'] = 'sale'
    elif 'for lease' in query_lower:
        entities['listing_type'] = 'lease'
    
    # Detect sale type
    if 'installment' in query_lower or 'installment plan' in query_lower or 'installment payment' in query_lower:
        entities['sale_type'] = 'installment'
        entities['financing_type'] = 'installment'
        logger.info(f"💰 Detected sale_type: installment")
    elif 'outright' in query_lower or 'cash' in query_lower or 'straight cash' in query_lower:
        entities['sale_type'] = 'outright'
        entities['financing_type'] = 'cash'
        logger.info(f"💰 Detected sale_type: outright")
    elif 'bank financing' in query_lower or 'bank loan' in query_lower or 'mortgage' in query_lower:
        entities['sale_type'] = 'bank_financing'
        entities['financing_type'] = 'bank_financing'
        logger.info(f"💰 Detected sale_type: bank_financing")
    
    # Detect specific financing options
    if 'bdo' in query_lower:
        entities['financing_options'] = 'BDO'
        entities['financing_type'] = 'bank_financing'
        logger.info(f"🏦 Detected financing_option: BDO")
    elif 'metrobank' in query_lower:
        entities['financing_options'] = 'Metrobank'
        entities['financing_type'] = 'bank_financing'
        logger.info(f"🏦 Detected financing_option: Metrobank")
    elif 'unionbank' in query_lower or 'union bank' in query_lower:
        entities['financing_options'] = 'UnionBank'
        entities['financing_type'] = 'bank_financing'
        logger.info(f"🏦 Detected financing_option: UnionBank")
    elif 'rcbc' in query_lower:
        entities['financing_options'] = 'RCBC'
        entities['financing_type'] = 'bank_financing'
        logger.info(f"🏦 Detected financing_option: RCBC")
    elif 'pag-ibig' in query_lower or 'pagibig' in query_lower:
        entities['financing_options'] = 'Pag-IBIG'
        entities['financing_type'] = 'pag_ibig'
        logger.info(f"🏦 Detected financing_option: Pag-IBIG")
    elif 'housing loan' in query_lower:
        entities['financing_options'] = 'Housing Loan'
        entities['financing_type'] = 'housing_loan'
        logger.info(f"🏦 Detected financing_option: Housing Loan")
    
    # Property type detection
    property_type_map = {
        'apartment': 'apartment',
        'apartments': 'apartment',
        'condo': 'condo', 'condominium': 'condo', 'condos': 'condo',
        'house': 'house', 'houses': 'house', 'villa': 'house', 'bungalow': 'house',
        'townhouse': 'townhouse', 'townhouses': 'townhouse',
        'commercial': 'commercial_building',
        'office': 'office_unit',
        'retail': 'retail_space',
        'warehouse': 'warehouse',
        'land': 'residential_lot', 'lot': 'residential_lot',
        'beachfront': 'beachfront',
        'resort': 'resort_property'
    }
    
    for key, value in property_type_map.items():
        if key in query_lower:
            entities['property_type'] = value
            break
    
    # Location detection
    batangas_locations = {
        'batangas city': 'Batangas City',
        'lipa': 'Lipa City', 'lipa city': 'Lipa City',
        'nasugbu': 'Nasugbu',
        'tanauan': 'Tanauan City', 'tanauan city': 'Tanauan City',
        'taal': 'Taal',
        'calatagan': 'Calatagan',
        'mabini': 'Mabini',
        'malvar': 'Malvar',
        'bauan': 'Bauan',
        'balayan': 'Balayan',
        'san juan': 'San Juan',
        'sto tomas': 'Sto. Tomas City', 'santo tomas': 'Sto. Tomas City',
        'sto. tomas': 'Sto. Tomas City',
        'tuy': 'Tuy', 'tuy batangas': 'Tuy',
        'lian': 'Lian', 'lian batangas': 'Lian',
        'taysan': 'Taysan', 'taysan batangas': 'Taysan',
        'rosario': 'Rosario', 'rosario batangas': 'Rosario',
        'laurel': 'Laurel',
        'agoncillo': 'Agoncillo',
        'san pascual': 'San Pascual',
        'cuenca': 'Cuenca',
        'alitagtag': 'Alitagtag',
        'san luis': 'San Luis',
        'padre garcia': 'Padre Garcia',
        'san nicolas': 'San Nicolas',
        'mataas na kahoy': 'Mataas Na Kahoy', 'mataasnakahoy': 'Mataas Na Kahoy',
        'talisay': 'Talisay',
        'la paz': 'La Paz',
        'lemery': 'Lemery',
        'ibaan': 'Ibaan',
        'lobo': 'Lobo',
        'tingloy': 'Tingloy'
    }
    
    for location_key, location_value in batangas_locations.items():
        if location_key in query_lower:
            entities['location'] = location_value
            has_specific_location = True
            break
        elif re.search(r'\b' + re.escape(location_key) + r'\b', query_lower):
            entities['location'] = location_value
            has_specific_location = True
            break
    
    # Feature detection
    if 'with swimming pool' in query_lower or 'with pool' in query_lower:
        entities['feature'] = 'swimming pool'
    elif 'with garden' in query_lower:
        entities['feature'] = 'garden'
    elif 'with parking' in query_lower:
        entities['feature'] = 'parking'
    elif 'furnished' in query_lower:
        entities['feature'] = 'furnished'
    
    # Landmark detection
    landmark_patterns = [
        (r'(?:near|close to|around|beside|next to|within walking distance of|walking distance to)\s+(?:the\s+)?(.+?)(?:\s+(?:in|at)\s+|$)', 'general'),
        (r'properties\s+(?:near|close to|around)\s+(?:the\s+)?(.+?)(?:\s+|$)', 'general'),
    ]
    
    for pattern, landmark_type in landmark_patterns:
        match = re.search(pattern, query_lower)
        if match:
            landmark = match.group(1).strip()
            entities['landmark'] = landmark
            entities['landmark_type'] = classify_landmark_type(landmark)
            entities['google_maps_check'] = True
            break
    
    # If no pattern match, try simple detection
    if not entities['landmark'] and 'near' in query_lower:
        parts = query_lower.split('near')
        if len(parts) > 1:
            landmark = parts[1].strip()
            entities['landmark'] = landmark
            entities['landmark_type'] = classify_landmark_type(landmark)
            entities['google_maps_check'] = True
    
    # General search detection
    if entities.get('property_type') and not has_specific_location:
        entities['has_general_search'] = True
        logger.info(f"🔍 Detected general search for {entities['property_type']} (no location specified)")
    
    logger.info(f"✅ Entities extracted: {entities}")
    return entities

# ========== LANDMARK PROXIMITY CHECKING ==========
def check_property_near_landmark(property_data: Dict[str, Any], landmark: str, landmark_type: str, api_key: str = None) -> Dict[str, Any]:
    """
    Check if a property is near a specific landmark using Google Maps API
    Returns: {'is_near': bool, 'distance': str, 'duration': str, 'landmark_found': str, 'match_type': str}
    """
    result = {
        'is_near': False,
        'distance': 'Not checked',
        'duration': 'Unknown',
        'landmark_found': None,
        'match_type': 'not_checked'
    }
    
    # If no API key or coordinates, fallback to description check
    if not api_key or 'latitude' not in property_data or 'longitude' not in property_data:
        return check_property_description_for_landmark(property_data, landmark, landmark_type)
    
    try:
        # Initialize Google Maps client
        gmaps = googlemaps.Client(key=api_key)
        
        # Property coordinates
        property_location = (property_data['latitude'], property_data['longitude'])
        
        # Search for the landmark near the property
        places_result = gmaps.places_nearby(
            location=property_location,
            keyword=landmark,
            radius=2000,  # 2km radius
            type='school' if landmark_type == 'school' else None
        )
        
        if places_result['results']:
            # Get the closest landmark
            closest = places_result['results'][0]
            landmark_location = (
                closest['geometry']['location']['lat'],
                closest['geometry']['location']['lng']
            )
            
            # Calculate walking distance
            distance_matrix = gmaps.distance_matrix(
                origins=[property_location],
                destinations=[landmark_location],
                mode='walking'
            )
            
            if distance_matrix['rows'][0]['elements'][0]['status'] == 'OK':
                element = distance_matrix['rows'][0]['elements'][0]
                distance = element['distance']['text']
                duration = element['duration']['text']
                
                # Consider "near" if within 2km walking distance
                distance_value = element['distance']['value']  # in meters
                result['is_near'] = distance_value <= 2000  # 2km
                result['distance'] = distance
                result['duration'] = duration
                result['landmark_found'] = closest['name']
                result['match_type'] = 'google_maps'
        
    except Exception as e:
        logger.error(f"❌ Google Maps API error: {e}")
        # Fallback to description check
        return check_property_description_for_landmark(property_data, landmark, landmark_type)
    
    return result

def check_property_description_for_landmark(property_data: Dict[str, Any], landmark: str, landmark_type: str) -> Dict[str, Any]:
    """
    Check property description and title for mentions of being near landmarks
    """
    result = {
        'is_near': False,
        'distance': 'Not mentioned',
        'landmark_found': None,
        'match_type': 'not_found'
    }
    
    description = property_data.get('description', '').lower()
    title = property_data.get('title', '').lower()
    address = property_data.get('address', '').lower()
    
    all_text = f"{title} {description} {address}"
    
    # Define comprehensive keyword sets
    if landmark_type == 'school':
        # Exact matches for specific schools mentioned
        if 'school' in landmark.lower():
            school_name = landmark.lower()
            if school_name in all_text:
                result['is_near'] = True
                result['distance'] = 'Exact match in description'
                result['landmark_found'] = landmark
                result['match_type'] = 'exact_match'
                return result
        
        # School proximity keywords (with confidence scores)
        proximity_keywords = [
            # Direct proximity mentions (high confidence)
            ('near school', 1.0),
            ('close to school', 1.0),
            ('walking distance to school', 1.0),
            ('proximity to school', 1.0),
            ('beside school', 1.0),
            ('adjacent to school', 1.0),
            ('next to school', 1.0),
            ('school nearby', 1.0),
            
            # General school area mentions (medium-high confidence)
            ('school district', 0.9),
            ('school zone', 0.9),
            ('school area', 0.8),
            ('near campus', 0.9),
            ('close to campus', 0.9),
            ('university area', 0.8),
            ('college town', 0.8),
            ('educational hub', 0.8),
            
            # Education-related mentions (medium confidence)
            ('educational institutions', 0.7),
            ('academic institutions', 0.7),
            ('learning centers', 0.6),
            ('good for students', 0.7),
            ('student-friendly', 0.7),
            ('family-friendly near schools', 0.8),
            ('education access', 0.6),
            
            # Specific school types (high confidence)
            ('near elementary school', 1.0),
            ('near high school', 1.0),
            ('near university', 1.0),
            ('near college', 1.0),
            ('near international school', 1.0),
            ('near private school', 1.0),
            ('near public school', 1.0),
            
            # Transportation to schools (medium confidence)
            ('short drive to school', 0.7),
            ('minutes to school', 0.7),
            ('accessible to schools', 0.7),
            ('school bus route', 0.6),
            ('school shuttle', 0.6),
            
            # Area benefits (medium confidence)
            ('good schools area', 0.8),
            ('school-friendly community', 0.7),
            ('near educational facilities', 0.8),
            ('academic environment', 0.6),
        ]
        
        # School name patterns
        school_patterns = [
            r'near\s+(\w+\s+school)',
            r'close\s+to\s+(\w+\s+school)',
            r'walking\s+distance\s+to\s+(\w+\s+school)',
            r'proximity\s+to\s+(\w+\s+school)',
            r'(\w+\s+school)\s+proximity',
            r'(\w+\s+university)\s+nearby',
            r'(\w+\s+college)\s+area',
            r'(\w+\s+academy)\s+near',
            r'(\w+\s+institute)\s+accessible',
        ]
        
        # Check for exact proximity mentions
        for keyword, confidence in proximity_keywords:
            if keyword in all_text:
                result['is_near'] = True
                result['distance'] = f'Mentioned: "{keyword}"'
                result['landmark_found'] = landmark
                result['match_type'] = 'proximity_keyword'
                result['confidence'] = confidence
                break
        
        # Check for school name patterns
        if not result['is_near']:
            for pattern in school_patterns:
                match = re.search(pattern, all_text)
                if match:
                    school_name_found = match.group(1)
                    result['is_near'] = True
                    result['distance'] = f'Near {school_name_found.title()}'
                    result['landmark_found'] = school_name_found.title()
                    result['match_type'] = 'school_pattern'
                    break
        
        # Check for general education mentions
        if not result['is_near']:
            education_keywords = [
                'education', 'academic', 'campus', 'university', 'college', 
                'student', 'tuition', 'faculty', 'scholar', 'learning',
                'study', 'teacher', 'professor', 'academy', 'institute'
            ]
            
            education_count = sum(1 for word in education_keywords if word in all_text)
            if education_count >= 3:
                result['is_near'] = True
                result['distance'] = 'Education-focused area'
                result['landmark_found'] = 'Educational institutions'
                result['match_type'] = 'education_general'
    
    # Similar logic for other landmark types
    elif landmark_type == 'hospital':
        hospital_keywords = [
            ('near hospital', 1.0),
            ('close to hospital', 1.0),
            ('medical center', 0.8),
            ('healthcare facilities', 0.7),
            ('accessible to hospital', 0.6),
            ('emergency services', 0.5),
            ('walking distance to hospital', 1.0),
        ]
        
        for keyword, confidence in hospital_keywords:
            if keyword in all_text:
                result['is_near'] = True
                result['distance'] = f'Mentioned: "{keyword}"'
                result['landmark_found'] = landmark
                result['match_type'] = 'proximity_keyword'
                break
    
    elif landmark_type == 'mall':
        mall_keywords = [
            ('near mall', 1.0),
            ('close to mall', 1.0),
            ('shopping center', 0.8),
            ('walking distance to mall', 1.0),
            ('accessible to mall', 0.7),
        ]
        
        for keyword, confidence in mall_keywords:
            if keyword in all_text:
                result['is_near'] = True
                result['distance'] = f'Mentioned: "{keyword}"'
                result['landmark_found'] = landmark
                result['match_type'] = 'proximity_keyword'
                break
    
    return result

def calculate_proximity_score(proximity_info: Dict[str, Any]) -> float:
    """Calculate a score based on how confidently we know the property is near the landmark"""
    score = 0.0
    
    # Base score for being near
    if proximity_info.get('is_near'):
        score += 1.0
    
    # Match type scoring
    match_type = proximity_info.get('match_type')
    match_type_scores = {
        'google_maps': 1.0,          # Direct Google Maps verification
        'exact_match': 0.95,         # Exact school name match
        'proximity_keyword': 0.9,    # Direct proximity mention
        'school_pattern': 0.85,      # School name pattern match
        'education_general': 0.7,    # General education mentions
        'not_found': 0.0,
        'not_checked': 0.0
    }
    
    if match_type in match_type_scores:
        score += match_type_scores[match_type]
    
    # Distance-based scoring (if available from API)
    distance = proximity_info.get('distance', '')
    if 'km' in distance:
        try:
            match = re.search(r'(\d+\.?\d*)\s*km', distance)
            if match:
                km = float(match.group(1))
                if km <= 0.5:
                    score += 1.0  # Very close (≤500m)
                elif km <= 1.0:
                    score += 0.8  # Close (≤1km)
                elif km <= 2.0:
                    score += 0.6  # Walking distance (≤2km)
                elif km <= 5.0:
                    score += 0.3  # Short drive
        except:
            pass
    elif 'm' in distance and 'km' not in distance:
        try:
            match = re.search(r'(\d+)\s*m', distance)
            if match:
                meters = float(match.group(1))
                if meters <= 500:
                    score += 1.0  # Very close
                elif meters <= 1000:
                    score += 0.8  # Close
                elif meters <= 2000:
                    score += 0.6  # Walking distance
        except:
            pass
    
    # Confidence score from keyword matching
    if 'confidence' in proximity_info:
        score += proximity_info['confidence']
    
    return min(score, 3.0)  # Cap at 3.0

def check_amenities_for_landmark(property_data: Dict[str, Any], landmark_type: str) -> bool:
    """Check if amenities list mentions schools or educational facilities"""
    amenities = property_data.get('amenities', [])
    
    if landmark_type == 'school':
        school_amenities = [
            'school', 'education', 'campus', 'university', 'college',
            'academic', 'student', 'learning', 'study', 'educational'
        ]
        
        for amenity in amenities:
            amenity_lower = str(amenity).lower()
            for keyword in school_amenities:
                if keyword in amenity_lower:
                    return True
    
    return False

# ========== HELPER FUNCTIONS ==========
def add_price_numeric_value(property_data: Dict) -> Dict:
    """Add numeric price value to property data for easier filtering"""
    property_data = property_data.copy()
    
    listing_type = property_data.get('type', property_data.get('listingType', 'unknown'))
    
    if listing_type == 'rent' and 'monthlyRent' in property_data:
        property_data['price_numeric'] = property_data['monthlyRent']
    elif listing_type == 'sale' and 'salePrice' in property_data:
        property_data['price_numeric'] = property_data['salePrice']
    elif listing_type == 'lease' and 'annualRent' in property_data:
        property_data['price_numeric'] = property_data['annualRent']
    else:
        price_str = str(property_data.get('price', '0'))
        try:
            match = re.search(r'[\d\.\,]+', price_str)
            if match:
                numeric_str = match.group().replace(',', '')
                if 'M' in price_str or 'm' in price_str:
                    property_data['price_numeric'] = float(numeric_str) * 1000000
                elif 'K' in price_str or 'k' in price_str:
                    property_data['price_numeric'] = float(numeric_str) * 1000
                else:
                    property_data['price_numeric'] = float(numeric_str)
            else:
                property_data['price_numeric'] = 0
        except:
            property_data['price_numeric'] = 0
    
    return property_data

def get_bedroom_count_from_string(bedroom_str: str) -> int:
    """Convert bedroom string to numeric count for filtering"""
    if not bedroom_str:
        return 0
    
    bedroom_str = str(bedroom_str).lower().strip()
    
    # Common patterns
    if bedroom_str == 'studio' or bedroom_str == '0' or 'studio' in bedroom_str:
        return 0
    elif bedroom_str == '1' or '1 bedroom' in bedroom_str:
        return 1
    elif bedroom_str == '2' or '2 bedroom' in bedroom_str:
        return 2
    elif bedroom_str == '3' or '3 bedroom' in bedroom_str:
        return 3
    elif bedroom_str == '4' or '4 bedroom' in bedroom_str:
        return 4
    elif bedroom_str == '5' or '5 bedroom' in bedroom_str or '5+' in bedroom_str:
        return 5
    elif '6' in bedroom_str or '6+' in bedroom_str:
        return 6
    else:
        # Try to extract any number
        match = re.search(r'(\d+)', bedroom_str)
        if match:
            return int(match.group(1))
        return 0

def calculate_family_suitability_score(prop: Dict[str, Any], family_size: int) -> int:
    """Calculate how suitable a property is for a family of given size"""
    score = 0
    
    # 1. Score based on property type (higher score for family-friendly types)
    prop_type = prop.get('propertyType', '').lower()
    prop_category = prop.get('propertyCategory', '').lower()
    
    # Property type scoring
    type_scores = {
        'house': 10,
        'townhouse': 9,
        'bungalow': 8,
        'duplex': 8,
        'condo': 6,
        'apartment': 5,
        'village_lot': 7,  # For building custom home
        'residential_lot': 7,
        'penthouse': 8,  # Spacious condo option
        'loft': 6,
        'boarding_house': 4,
        'room': 2,
        'dormitory': 3,
    }
    
    for type_key, type_score in type_scores.items():
        if type_key in prop_type or type_key in str(prop.get('type', '')).lower():
            score += type_score
            break
    
    # 2. Score based on bedrooms
    bedroom_str = prop.get('bedrooms', '')
    bedroom_count = get_bedroom_count_from_string(bedroom_str)
    
    if bedroom_count > 0:
        # Ideal bedroom count based on family size
        if family_size <= 2:
            ideal_bedrooms = 2
        elif family_size <= 4:
            ideal_bedrooms = 3
        else:  # 5+ members
            ideal_bedrooms = 4
        
        if bedroom_count >= ideal_bedrooms:
            score += 15  # Meets or exceeds ideal
        elif bedroom_count >= ideal_bedrooms - 1:
            score += 10  # Close to ideal
        else:
            score += 5   # Less than ideal but might work
    
    # 3. Score based on bathrooms
    bathroom_str = prop.get('bathrooms', '')
    if bathroom_str:
        try:
            if bathroom_str == '4+':
                bathroom_count = 4
            else:
                bathroom_count = int(bathroom_str)
            
            if bathroom_count >= 2:
                score += 8  # Multiple bathrooms are great for families
            elif bathroom_count >= 1:
                score += 4
        except:
            pass
    
    # 4. Score based on space/size
    floor_area = prop.get('floorArea', 0)
    lot_area = prop.get('lotArea', 0)
    
    if floor_area > 0:
        if floor_area >= 100:  # 100+ sqm is spacious for families
            score += 10
        elif floor_area >= 60:  # 60-99 sqm is decent
            score += 6
        elif floor_area >= 30:  # 30-59 sqm is basic
            score += 3
    
    if lot_area and lot_area > 100:  # Large lot is great for families
        score += 8
    
    # 5. Score based on amenities/features
    amenities = prop.get('amenities', [])
    features = prop.get('features', [])
    all_features = amenities + features
    
    family_friendly_features = {
        'garden': 5,
        'yard': 5,
        'parking': 4,
        'parking space': 4,
        'spacious': 3,
        'children': 3,
        'family': 3,
        'playground': 6,
        'pool': 4,
        'swimming pool': 4,
        'security': 3,
        'fenced': 3,
        'safe': 3,
        'quiet': 2,
        'community': 2,
        'near school': 6,
        'near park': 4,
        'school proximity': 5,
        'multiple bathrooms': 4,
        'storage': 2,
        'laundry': 2,
    }
    
    for feature in all_features:
        feature_lower = str(feature).lower()
        for key, value in family_friendly_features.items():
            if key in feature_lower:
                score += value
    
    # 6. Score based on location/neighborhood
    description = prop.get('description', '').lower()
    title = prop.get('title', '').lower()
    
    location_keywords = ['family-friendly', 'safe neighborhood', 'quiet street', 
                        'good for families', 'child-friendly', 'residential area',
                        'subdivision', 'village']
    
    for keyword in location_keywords:
        if keyword in description or keyword in title:
            score += 4
    
    # 7. Score based on furnishing
    furnishing = prop.get('furnishing', '').lower()
    if 'furnished' in furnishing:
        score += 3  # Helpful for families moving in
    elif 'semi-furnished' in furnishing:
        score += 2
    
    return score
    
def generate_family_needs_response(family_size: int, properties: List[Dict[str, Any]], entities: Dict[str, Any]) -> str:
    """Generate response specifically for family needs"""
    
    if not properties:
        return f"I couldn't find any properties for family of {family_size} members.\n\n"
    
    # Calculate ideal bedroom range based on family size
    if family_size <= 2:
        ideal_min, ideal_max = 1, 2  # 1-2 bedrooms ideal for couple/small family
    elif 3 <= family_size <= 4:
        ideal_min, ideal_max = 2, 3  # 2-3 bedrooms ideal for small family
    elif 5 <= family_size <= 6:
        ideal_min, ideal_max = 3, 4  # 3-4 bedrooms ideal for medium family
    else:  # 7+ people
        ideal_min, ideal_max = 4, 5  # 4+ bedrooms ideal for large family
    
    # Score and sort properties based on bedroom suitability
    scored_properties = []
    
    for prop in properties:
        # Skip obviously unsuitable property types
        prop_type = str(prop.get('propertyType', '')).lower()
        unsuitable_types = ['room', 'boarding_house', 'dormitory', 'office', 
                           'retail', 'commercial', 'warehouse', 'industrial',
                           'food_stall', 'shop', 'showroom', 'parking_area']
        
        if any(unsuitable in prop_type for unsuitable in unsuitable_types):
            continue
            
        # Get bedroom count
        bedroom_str = prop.get('bedrooms', '')
        bedrooms = get_bedroom_count_from_string(bedroom_str)
        
        # Calculate bedroom suitability score
        bedroom_score = 0
        if ideal_min <= bedrooms <= ideal_max:
            bedroom_score = 100  # Perfect match!
        elif bedrooms == ideal_max + 1:
            bedroom_score = 80   # Slightly larger than ideal
        elif bedrooms == ideal_max + 2:
            bedroom_score = 60   # Much larger than ideal
        elif bedrooms == ideal_min - 1 and bedrooms > 0:
            bedroom_score = 70   # Slightly smaller than ideal
        elif bedrooms < ideal_min:
            bedroom_score = 50   # Smaller than minimum
        elif bedrooms > ideal_max + 2:
            bedroom_score = 40   # Much larger than ideal
            
        # Calculate overall family suitability
        overall_score = calculate_family_suitability_score(prop, family_size)
        
        # Combine scores (bedroom match is more important)
        total_score = (bedroom_score * 0.6) + (overall_score * 0.4)
        
        prop['family_suitability_score'] = total_score
        prop['bedroom_match_score'] = bedroom_score
        scored_properties.append(prop)
    
    if not scored_properties:
        return f"I couldn't find any suitable properties for family of {family_size} members.\n\n"
    
    # Sort by total suitability score (highest first)
    scored_properties.sort(key=lambda x: x.get('family_suitability_score', 0), reverse=True)
    
    # Take top 5 most suitable properties
    filtered_properties = scored_properties[:5]
    
    # Generate response
    response = f"🏠 **Properties Suitable for Family of {family_size}**\n\n"
    
    if filtered_properties:
        response += f"I found {len(filtered_properties)} properties that could work for your family:\n\n"
        
        for i, prop in enumerate(filtered_properties):
            title = prop.get('title', f'Property {i+1}')
            price = prop.get('price', 'Price not available')
            location = prop.get('location', 'Location not specified')
            bedrooms = prop.get('bedrooms', 'Not specified')
            prop_type = prop.get('type', '').replace('_', ' ').title()
            bedroom_score = prop.get('bedroom_match_score', 0)
            
            response += f"{i+1}. **{title}**\n"
            response += f"   📍 {location}\n"
            response += f"   🏠 Type: {prop_type}\n"
            response += f"   💰 Price: {price}\n"
            response += f"   🛏️ Bedrooms: {bedrooms}"
            
            # Add bedroom suitability note
            if bedroom_score == 100:
                response += f" ✅ **Perfect size for your family**\n"
            elif bedroom_score >= 80:
                response += f" 👍 **Good size for your family**\n"
            elif bedroom_score >= 60:
                response += f" 📏 **Adequate size for your family**\n"
            else:
                response += f" 📐 **May need adjustment for your family**\n"
            
            # Show key features
            features = prop.get('features', [])
            if features:
                response += f"   ✅ Features: {', '.join(features[:3])}\n"
            
            response += "\n"
        
        # Family living tips based on size
        response += "**💡 Family Living Tips:**\n"
        
        if family_size == 1:
            response += "• **Studio or 1 bedroom** is perfect for singles\n"
            response += "• Consider **condos or apartments** for low maintenance\n"
            response += "• Look for **secure buildings** with amenities\n"
            
        elif family_size == 2:  # Couple or 1 child
            response += "• **1-2 bedroom properties** provide space for home office or guest room\n"
            response += "• Look for **secure buildings** or **gated communities**\n"
            response += "• Consider **proximity to schools** even if no children yet\n"
            
        elif family_size == 3:  # Small family
            response += "• **2-3 bedroom properties** are ideal for growing families\n"
            response += "• Multiple **bathrooms** help with morning routines\n"
            response += "• **Nearby parks and playgrounds** are great for children\n"
            
        elif family_size == 4:  # Standard family
            response += "• **3-4 bedroom properties** provide comfortable living space\n"
            response += "• **2+ bathrooms** are recommended for convenience\n"
            response += "• **Yard or garden space** allows for outdoor activities\n"
            
        elif family_size >= 5:  # Large family
            response += "• **4+ bedroom properties** or **houses with extension potential**\n"
            response += "• **Multiple living areas** help with space management\n"
            response += "• **Large lots** allow for expansion or outdoor space\n"
        
    else:
        response = f"I couldn't find specifically family-optimized properties for {family_size} members.\n\n"
        response += "💡 **Try these adjustments:**\n"
        min_bedrooms = entities.get('min_bedrooms', 2)
        response += f"• Search for properties with **{min_bedrooms}+ bedrooms**\n"
        response += "• Look in family-friendly areas like subdivisions\n"
        response += "• Consider properties with 'family-friendly' features\n\n"
        
        response += "**🔍 Try these specific searches:**\n"
        response += f"• *'find houses with {min_bedrooms} bedrooms'*\n"
        response += "• *'show me properties with garden for families'*\n"
        response += "• *'properties in gated communities'*\n"
    
    return response

def calculate_installment_payment(property_data: Dict) -> Optional[Dict]:
    """Calculate installment payment details for a property"""
    sale_price = property_data.get('salePrice')
    if not sale_price or sale_price <= 0:
        return None
    
    downpayment_percentage = 0.30
    loan_term_years = 5
    annual_interest_rate = 0.06
    
    downpayment = sale_price * downpayment_percentage
    loan_amount = sale_price - downpayment
    
    total_interest = loan_amount * annual_interest_rate * loan_term_years
    total_payment = loan_amount + total_interest
    monthly_payment = total_payment / (loan_term_years * 12)
    
    return {
        'downpayment': round(downpayment, 2),
        'loan_amount': round(loan_amount, 2),
        'monthly_payment': round(monthly_payment, 2),
        'interest_rate': f"{annual_interest_rate * 100}%",
        'term_years': loan_term_years,
        'total_payment': round(total_payment, 2)
    }

def standardize_property_data(property_data: Dict) -> Dict:
    """Standardize property data from Firestore to chatbot format"""
    title = property_data.get('title', 'Untitled Property')
    property_type = property_data.get('propertyType', property_data.get('type', 'unknown'))
    city = property_data.get('city', 'Unknown')
    province = property_data.get('province', 'Batangas')
    
    listing_type = property_data.get('type', property_data.get('listingType', 'unknown'))
    price_str = "Price not available"
    
    if listing_type == 'rent' and 'monthlyRent' in property_data:
        price = property_data['monthlyRent']
        price_str = f"₱{price:,.0f}/month"
    elif listing_type == 'sale' and 'salePrice' in property_data:
        price = property_data['salePrice']
        if price >= 1000000:
            price_str = f"₱{price/1000000:.1f}M"
        else:
            price_str = f"₱{price:,.0f}"
    elif listing_type == 'lease' and 'annualRent' in property_data:
        price = property_data['annualRent']
        price_str = f"₱{price:,.0f}/year"
    
    features = []
    if property_data.get('furnishing'):
        features.append(property_data['furnishing'])
    if property_data.get('amenities'):
        features.extend(property_data['amenities'][:3])
    if property_data.get('bedrooms'):
        features.append(f"{property_data['bedrooms']} bedroom{'s' if property_data['bedrooms'] != '1' else ''}")
    if property_data.get('bathrooms'):
        features.append(f"{property_data['bathrooms']} bathroom{'s' if property_data['bathrooms'] != '1' else ''}")
    
    description = property_data.get('description', '')
    if not description:
        description = f"A {property_type.replace('_', ' ')} located in {city}, {province}."
    
    standardized = {
        'id': property_data.get('id', ''),
        'title': title,
        'type': property_type,
        'location': f"{city}, {province}",
        'city': city,
        'province': province,
        'price': price_str,
        'bedrooms': property_data.get('bedrooms', 'Not specified'),
        'bathrooms': property_data.get('bathrooms', 'Not specified'),
        'features': features,
        'description': description,
        'listing_type': listing_type,
        'status': property_data.get('status', 'unknown'),
        'address': property_data.get('address', ''),
        'imageUrls': property_data.get('imageUrls', []) or property_data.get('photos', []),
        'videoUrls': property_data.get('videoUrls', []),
        'hasVideos': property_data.get('hasVideos', False),
        'floorArea': property_data.get('floorArea', None),
        'lotArea': property_data.get('lotArea', None),
        'financingOptions': property_data.get('financingOptions', []),
        'saleType': property_data.get('saleType', 'Not specified'),
        'salePrice': property_data.get('salePrice', 0),
        'price_numeric': property_data.get('price_numeric', 0),
        'latitude': property_data.get('latitude', None),
        'longitude': property_data.get('longitude', None),
        'amenities': property_data.get('amenities', [])
    }
    
    if property_data.get('installment_details'):
        standardized['installment_details'] = property_data['installment_details']
    
    return standardized

def get_mock_properties(entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate mock properties for testing when Firebase is not connected"""
    mock_properties = []
    
    base_properties = [
        {
            'id': 'mock_1',
            'title': 'Modern House Near Schools in Nasugbu',
            'propertyType': 'house',
            'type': 'rent',
            'city': 'Nasugbu',
            'province': 'Batangas',
            'address': '123 Beach Road, Nasugbu',
            'monthlyRent': 25000,
            'bedrooms': '3',
            'bathrooms': '2',
            'floorArea': 120,
            'description': 'Beautiful modern house near Nasugbu Elementary School. Walking distance to schools and perfect for families. Close to educational institutions.',
            'imageUrls': [],
            'status': 'available',
            'amenities': ['Swimming Pool', 'Garden', 'Parking', 'Near Schools'],
            'latitude': 14.0735,
            'longitude': 120.6354
        },
        {
            'id': 'mock_2',
            'title': 'Beachfront Condo Unit with School Proximity',
            'propertyType': 'condo',
            'type': 'sale',
            'saleType': 'installment',
            'city': 'Nasugbu',
            'province': 'Batangas',
            'address': '456 Coastal Avenue, Nasugbu',
            'salePrice': 3500000,
            'bedrooms': '2',
            'bathrooms': '2',
            'floorArea': 80,
            'description': 'Luxury beachfront condo with ocean view. Near Nasugbu National High School. Educational facilities accessible within 5 minutes.',
            'imageUrls': [],
            'status': 'available',
            'financingOptions': ['Bank Financing - BDO', 'Pag-IBIG Housing Loan'],
            'amenities': ['Pool', 'Gym', '24/7 Security'],
            'latitude': 14.0750,
            'longitude': 120.6360
        },
        {
            'id': 'mock_3',
            'title': 'Commercial Space in Lipa Near Universities',
            'propertyType': 'commercial_building',
            'type': 'lease',
            'city': 'Lipa City',
            'province': 'Batangas',
            'address': '789 Business District, Lipa',
            'annualRent': 1200000,
            'description': 'Prime commercial space for business. Located near University of Batangas Lipa Campus. Student-friendly area with good foot traffic.',
            'imageUrls': [],
            'status': 'available',
            'latitude': 13.9410,
            'longitude': 121.1630
        },
        {
            'id': 'mock_4',
            'title': 'Apartment Near Batangas State University',
            'propertyType': 'apartment',
            'type': 'rent',
            'city': 'Batangas City',
            'province': 'Batangas',
            'address': '101 Main Street, Batangas City',
            'monthlyRent': 12000,
            'bedrooms': '2',
            'bathrooms': '1',
            'floorArea': 50,
            'description': 'Clean and affordable apartment. Walking distance to Batangas State University. Perfect for students and young professionals.',
            'imageUrls': [],
            'status': 'available',
            'amenities': ['WiFi', 'Laundry Area', 'Study Area'],
            'latitude': 13.7565,
            'longitude': 121.0583
        },
        {
            'id': 'mock_5',
            'title': 'Townhouse in School District Sto. Tomas',
            'propertyType': 'townhouse',
            'type': 'sale',
            'saleType': 'bank_financing',
            'city': 'Sto. Tomas City',
            'province': 'Batangas',
            'address': '202 Subdivision, Sto. Tomas',
            'salePrice': 2800000,
            'bedrooms': '3',
            'bathrooms': '2',
            'floorArea': 90,
            'description': 'Modern townhouse with garage. Located in established school district. Near Sto. Tomas Elementary School and High School.',
            'imageUrls': [],
            'status': 'available',
            'financingOptions': ['Bank Financing - Metrobank', 'Outright Payment'],
            'amenities': ['Garage', 'Garden', 'Security'],
            'latitude': 14.0735,
            'longitude': 121.1410
        },
        {
            'id': 'mock_6',
            'title': 'Family House with School Access in Lipa',
            'propertyType': 'house',
            'type': 'sale',
            'saleType': 'installment',
            'city': 'Lipa City',
            'province': 'Batangas',
            'address': '303 Family Subdivision, Lipa',
            'salePrice': 4500000,
            'bedrooms': '4',
            'bathrooms': '3',
            'floorArea': 150,
            'description': 'Spacious family house with installment payment option. School bus route passes nearby. Educational hub with multiple schools in the area.',
            'imageUrls': [],
            'status': 'available',
            'financingOptions': ['In-house Installment Plan', 'Bank Financing - UnionBank'],
            'amenities': ['Large Yard', 'Parking for 2', 'Study Room'],
            'latitude': 13.9440,
            'longitude': 121.1620
        },
        {
            'id': 'mock_7',
            'title': 'Student Dormitory Near Campus',
            'propertyType': 'dormitory',
            'type': 'rent',
            'city': 'Lipa City',
            'province': 'Batangas',
            'address': '404 Student Village, Lipa',
            'monthlyRent': 8000,
            'bedrooms': '1',
            'bathrooms': '1',
            'floorArea': 25,
            'description': 'Student dormitory with all amenities. Right beside University of Batangas. Perfect for college students.',
            'imageUrls': [],
            'status': 'available',
            'amenities': ['WiFi', 'Study Room', 'Laundry', 'Cafeteria'],
            'latitude': 13.9430,
            'longitude': 121.1610
        }
    ]
    
    for prop in base_properties:
        matches = True
        
        if entities.get('location'):
            location = entities['location'].lower()
            prop_city = prop.get('city', '').lower()
            if 'nasugbu' in location and 'nasugbu' not in prop_city:
                matches = False
            elif 'lipa' in location and 'lipa' not in prop_city:
                matches = False
            elif 'batangas city' in location and 'batangas city' not in prop_city:
                matches = False
            elif 'sto tomas' in location and 'sto. tomas city' not in prop_city:
                matches = False
        
        if entities.get('property_type') and matches:
            requested_type = entities['property_type'].lower()
            prop_type = prop.get('propertyType', '').lower()
            
            type_mapping = {
                'house': ['house', 'bungalow', 'duplex'],
                'condo': ['condo', 'condominium', 'penthouse', 'studio'],
                'apartment': ['apartment', 'room', 'boarding_house', 'dormitory'],
                'commercial': ['commercial', 'office', 'retail', 'warehouse'],
                'townhouse': ['townhouse']
            }
            
            if requested_type in type_mapping:
                if prop_type not in type_mapping[requested_type]:
                    matches = False
        
        if entities.get('sale_type') and matches:
            prop_type = prop.get('type', '')
            prop_sale_type = prop.get('saleType', '')
            
            if prop_type == 'sale':
                if prop_sale_type != entities['sale_type']:
                    matches = False
            else:
                matches = False
        
        if entities.get('max_price') and matches:
            price_numeric = 0
            if prop.get('type') == 'rent' and 'monthlyRent' in prop:
                price_numeric = prop['monthlyRent']
            elif prop.get('type') == 'sale' and 'salePrice' in prop:
                price_numeric = prop['salePrice']
            
            if price_numeric > entities['max_price']:
                matches = False
        
        # Check for bedroom requirements
        if entities.get('exact_bedrooms') is not None and matches:
            prop_bedrooms = prop.get('bedrooms', 'Not specified')
            prop_bed_num = get_bedroom_count_from_string(prop_bedrooms)
            
            if prop_bed_num != entities['exact_bedrooms']:
                matches = False
                logger.debug(f"❌ Exact bedroom mismatch: {prop_bed_num} != {entities['exact_bedrooms']}")

        # Check for minimum bedroom requirement (for family needs)
        elif entities.get('min_bedrooms') is not None and matches:
            prop_bedrooms = prop.get('bedrooms', 'Not specified')
            prop_bed_num = get_bedroom_count_from_string(prop_bedrooms)
            
            if prop_bed_num < entities['min_bedrooms']:
                matches = False
                logger.debug(f"❌ Minimum bedroom requirement not met: {prop_bed_num} < {entities['min_bedrooms']}")
            else:
                logger.debug(f"✅ Meets minimum bedroom requirement: {prop_bed_num} >= {entities['min_bedrooms']}")
        
        if matches:
            prop_with_price = add_price_numeric_value(prop)
            
            if entities.get('sale_type') == 'installment' and prop.get('type') == 'sale':
                installment_details = calculate_installment_payment(prop_with_price)
                if installment_details:
                    prop_with_price['installment_details'] = installment_details
            
            mock_properties.append(standardize_property_data(prop_with_price))
    
    return mock_properties

def search_firestore_properties(entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Search properties in Firestore based on entities"""
    properties = []
    
    if not db:
        logger.warning("⚠️ Firebase not connected, returning mock data")
        return get_mock_properties(entities)
    
    try:
        properties_ref = db.collection('properties')
        query = properties_ref
        
        logger.info("🔍 Status filtering will be done client-side")
        
        # Filter by sale type
        sale_type = entities.get('sale_type')
        logger.info(f"🔍 Looking for sale_type: {sale_type}")
        
        if sale_type:
            try:
                query = query.where(filter=FieldFilter('type', '==', 'sale'))
                logger.info("✅ Filtered by type: sale")
                
                query = query.where(filter=FieldFilter('saleType', '==', sale_type))
                logger.info(f"✅ Filtered by saleType: {sale_type}")
            except Exception as e:
                logger.warning(f"⚠️ Could not filter by saleType: {e}")
        
        # Filter by specific financing options
        if entities.get('financing_options') and not sale_type:
            financing_option = entities['financing_options']
            logger.info(f"🔍 Looking for {financing_option} financing...")
            
            financing_map = {
                'BDO': ['BDO', 'Bank Financing - BDO', 'BDO Bank'],
                'Metrobank': ['Metrobank', 'Bank Financing - Metrobank', 'Metrobank Bank'],
                'UnionBank': ['UnionBank', 'Bank Financing - UnionBank', 'Union Bank'],
                'RCBC': ['RCBC', 'Bank Financing - RCBC', 'RCBC Bank'],
                'Pag-IBIG': ['Pag-IBIG', 'Pag-IBIG Housing Loan', 'Pag-IBIG Loan', 'Pagibig'],
                'Housing Loan': ['Housing Loan', 'Home Loan', 'Property Loan']
            }
            
            if financing_option in financing_map:
                search_terms = financing_map[financing_option]
                logger.info(f"🔍 Search terms for {financing_option}: {search_terms}")
                
                for term in search_terms:
                    try:
                        temp_query = query.where(filter=FieldFilter('financingOptions', 'array_contains', term))
                        test_docs = list(temp_query.limit(1).get())
                        if test_docs:
                            query = temp_query
                            logger.info(f"✅ Found properties with financing term: {term}")
                            break
                    except Exception as e:
                        logger.debug(f"⚠️ Could not search for {term}: {e}")
                        continue
        
        # Filter by location
        if entities.get('location'):
            location = entities['location']
            
            location_map = {
                'Batangas City': 'Batangas City',
                'Lipa City': 'Lipa City',
                'Nasugbu': 'Nasugbu',
                'Malvar': 'Malvar',
                'Mataas Na Kahoy': 'Mataas Na Kahoy',
                'Tanauan City': 'Tanauan City',
                'Taal': 'Taal',
                'Calatagan': 'Calatagan',
                'Mabini': 'Mabini',
                'Bauan': 'Bauan',
                'Balayan': 'Balayan',
                'San Juan': 'San Juan',
                'Sto. Tomas City': 'Sto. Tomas City',
                'Santo Tomas': 'Sto. Tomas City',
                'Sto Tomas': 'Sto. Tomas City'
            }
            
            if location in location_map:
                query = query.where(filter=FieldFilter('city', '==', location_map[location]))
                logger.info(f"🔍 Filtering by city: {location_map[location]}")
        
        # Filter by property type
        if entities.get('property_type'):
            property_type = entities['property_type']
            
            type_map = {
                'apartment': 'apartment',
                'apartments': 'apartment',
                'condo': 'condo_unit',
                'condos': 'condo_unit',
                'condominium': 'condo_unit',
                'condominiums': 'condo_unit',
                'house': 'house',
                'houses': 'house',
                'townhouse': 'townhouse',
                'townhouses': 'townhouse',
                'commercial': 'commercial_building',
                'commercial_space': 'commercial_building',
                'office': 'office_unit',
                'retail': 'retail_space',
                'warehouse': 'warehouse',
                'industrial': 'warehouse',
                'land': 'residential_lot',
                'lot': 'residential_lot',
                'residential_lot': 'residential_lot',
                'commercial_lot': 'commercial_lot',
                'agricultural': 'agricultural_land',
                'agricultural_land': 'agricultural_land',
                'beachfront': 'beachfront',
                'resort': 'resort_property',
                'resort_property': 'resort_property',
                'commercial_building': 'commercial_building',
                'office_unit': 'office_unit',
                'retail_space': 'retail_space'
            }
            
            mapped_type = type_map.get(property_type, property_type)
            try:
                query = query.where(filter=FieldFilter('propertyType', '==', mapped_type))
                logger.info(f"🔍 Filtering by property type: {mapped_type}")
            except Exception as e:
                logger.warning(f"⚠️ Could not filter by property type {mapped_type}: {e}")
        
        # Apply price filters
        if entities.get('max_price'):
            max_price = entities['max_price']
            logger.info(f"💰 Applying max price filter: ₱{max_price:,.0f}")
            
            price_fields = ['salePrice', 'monthlyRent', 'annualRent']
            price_filter_applied = False
            
            for field in price_fields:
                try:
                    query = query.where(filter=FieldFilter(field, '<=', max_price))
                    logger.info(f"🔍 Filtering by max {field}: ₱{max_price:,.0f}")
                    price_filter_applied = True
                    break
                except Exception as price_error:
                    logger.debug(f"⚠️ Could not filter by {field}: {price_error}")
                    continue
            
            if not price_filter_applied:
                logger.info("💡 Will apply price filtering client-side")
        
        # Execute query
        limit_count = 50  # Get more for landmark filtering
        logger.info(f"🔍 Executing Firestore query (limit: {limit_count})...")
        docs = query.limit(limit_count).get()
        
        property_data_list = []
        status_counts = {}
        
        for doc in docs:
            property_data = doc.to_dict()
            property_data['id'] = doc.id
            property_data_list.append(property_data)
            
            status = property_data.get('status', 'NO STATUS')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info(f"🔍 Found {len(property_data_list)} properties from Firestore")
        logger.info(f"🔍 Status breakdown: {status_counts}")
        
        # Client-side filtering
        filtered_properties = []
        
        for property_data in property_data_list:
            matches = True
            
            status = str(property_data.get('status', '')).lower()
            valid_statuses = ['available', 'active', 'for rent', 'for sale', 'for lease', 'listed']
            if status not in valid_statuses:
                logger.debug(f"❌ Property {property_data.get('id', 'unknown')} excluded - status: {status}")
                matches = False
                continue
            
            if sale_type and matches:
                prop_type = property_data.get('type', property_data.get('listingType', ''))
                prop_sale_type = property_data.get('saleType', '').lower()
                
                if prop_type != 'sale' or prop_sale_type != sale_type:
                    logger.debug(f"❌ Sale type mismatch: {prop_type}/{prop_sale_type} != sale/{sale_type}")
                    matches = False
            
            if entities.get('financing_options') and matches:
                financing_options = property_data.get('financingOptions', [])
                search_term = entities['financing_options'].lower()
                
                has_financing = False
                for option in financing_options:
                    if isinstance(option, str) and search_term in option.lower():
                        has_financing = True
                        break
                
                if not has_financing and sale_type != 'installment':
                    matches = False
                    logger.debug(f"❌ No {entities['financing_options']} financing found: {financing_options}")
            
            if not matches:
                continue
            
            property_data_with_price = add_price_numeric_value(property_data)
            
            if entities.get('max_price') and matches:
                price_numeric = property_data_with_price.get('price_numeric', 0)
                if price_numeric > entities['max_price']:
                    matches = False
                    logger.debug(f"❌ Price too high: {price_numeric} > {entities['max_price']}")
            
            if entities.get('min_price') and matches:
                price_numeric = property_data_with_price.get('price_numeric', 0)
                if price_numeric < entities['min_price']:
                    matches = False
                    logger.debug(f"❌ Price too low: {price_numeric} < {entities['min_price']}")
            
            # Check bedroom requirements
            if (entities.get('exact_bedrooms') is not None or 
                entities.get('min_bedrooms') is not None) and matches:
                
                prop_bedrooms = property_data.get('bedrooms', 'Not specified')
                try:
                    # Use the helper function to convert bedroom string to number
                    prop_bed_num = get_bedroom_count_from_string(prop_bedrooms)
                    
                    if entities.get('exact_bedrooms') is not None:
                        # Exact bedroom requirement
                        if prop_bed_num != entities['exact_bedrooms']:
                            matches = False
                            logger.debug(f"❌ Exact bedroom mismatch: {prop_bed_num} != {entities['exact_bedrooms']}")
                    
                    elif entities.get('min_bedrooms') is not None:
                        # Minimum bedroom requirement (for family needs)
                        if prop_bed_num < entities['min_bedrooms']:
                            matches = False
                            logger.debug(f"❌ Minimum bedroom requirement not met: {prop_bed_num} < {entities['min_bedrooms']}")
                        else:
                            logger.debug(f"✅ Meets minimum bedroom requirement: {prop_bed_num} >= {entities['min_bedrooms']}")
                            
                except Exception as e:
                    logger.debug(f"⚠️ Could not parse bedrooms: {e}")
                    # If we can't parse bedrooms but have a requirement, be conservative
                    if entities.get('min_bedrooms') is not None:
                        matches = False
            
            if matches:
                if sale_type == 'installment':
                    installment_details = calculate_installment_payment(property_data_with_price)
                    if installment_details:
                        property_data_with_price['installment_details'] = installment_details
                
                standardized_property = standardize_property_data(property_data_with_price)
                filtered_properties.append(standardized_property)
        
        properties = filtered_properties
        logger.info(f"🔍 After client-side filtering: {len(properties)} properties")
        
        # NEW: Enhanced filtering for landmark queries
        if entities.get('landmark') and entities.get('google_maps_check'):
            filtered_by_proximity = []
            google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
            landmark = entities['landmark']
            landmark_type = entities['landmark_type']
            
            logger.info(f"📍 Filtering properties near {landmark} (type: {landmark_type})")
            
            for prop in properties:
                has_proximity = False
                proximity_info = {}
                
                # Method 1: Google Maps API check (if coordinates available)
                if ('latitude' in prop and 'longitude' in prop and 
                    google_maps_api_key and GOOGLE_MAPS_AVAILABLE):
                    api_result = check_property_near_landmark(
                        prop, 
                        landmark, 
                        landmark_type,
                        google_maps_api_key
                    )
                    
                    if api_result['is_near']:
                        has_proximity = True
                        proximity_info = api_result
                        logger.info(f"✅ Google Maps: Property '{prop.get('title')}' is near {landmark}")
                
                # Method 2: Enhanced description check (always performed as backup)
                if not has_proximity:
                    desc_result = check_property_description_for_landmark(prop, landmark, landmark_type)
                    if desc_result['is_near']:
                        has_proximity = True
                        proximity_info = desc_result
                        logger.info(f"✅ Description: Property '{prop.get('title')}' mentions {landmark_type}")
                
                # Method 3: Check amenities list for schools/landmarks
                if not has_proximity and landmark_type == 'school':
                    if check_amenities_for_landmark(prop, landmark_type):
                        has_proximity = True
                        proximity_info = {
                            'is_near': True,
                            'distance': 'Listed as amenity',
                            'landmark_found': landmark,
                            'match_type': 'amenity'
                        }
                        logger.info(f"✅ Amenity: Property '{prop.get('title')}' has school-related amenity")
                
                # If property is near the landmark, add it to results
                if has_proximity:
                    prop['proximity_info'] = proximity_info
                    prop['proximity_score'] = calculate_proximity_score(proximity_info)
                    filtered_by_proximity.append(prop)
            
            # Sort by proximity score (highest first)
            filtered_by_proximity.sort(key=lambda x: x.get('proximity_score', 0), reverse=True)
            properties = filtered_by_proximity
            
            logger.info(f"📍 After proximity filtering: {len(properties)} properties near {landmark}")
        
        # Fallback for no results
        if len(properties) == 0 and sale_type:
            logger.info("🔄 No exact matches found, trying fallback search...")
            
            fallback_query = properties_ref.where(filter=FieldFilter('type', '==', 'sale'))
            
            if entities.get('location'):
                location = entities['location']
                location_map = {
                    'Batangas City': 'Batangas City',
                    'Lipa City': 'Lipa City',
                    'Nasugbu': 'Nasugbu',
                    'Sto. Tomas City': 'Sto. Tomas City',
                }
                if location in location_map:
                    fallback_query = fallback_query.where(filter=FieldFilter('city', '==', location_map[location]))
            
            fallback_docs = fallback_query.limit(10).get()
            
            for doc in fallback_docs:
                property_data = doc.to_dict()
                property_data['id'] = doc.id
                
                prop_sale_type = property_data.get('saleType', '').lower()
                if prop_sale_type == sale_type:
                    property_data_with_price = add_price_numeric_value(property_data)
                    
                    if sale_type == 'installment':
                        installment_details = calculate_installment_payment(property_data_with_price)
                        if installment_details:
                            property_data_with_price['installment_details'] = installment_details
                    
                    standardized_property = standardize_property_data(property_data_with_price)
                    properties.append(standardized_property)
            
            logger.info(f"🔄 Found {len(properties)} properties in fallback search")
        
        # Deduplication
        unique_properties = []
        seen_ids = set()
        
        for prop in properties:
            prop_id = prop.get('id')
            if prop_id and prop_id not in seen_ids:
                seen_ids.add(prop_id)
                unique_properties.append(prop)
        
        properties = unique_properties
        logger.info(f"🔍 After deduplication: {len(properties)} unique properties")
        
    except Exception as e:
        logger.error(f"❌ Error searching Firestore: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        properties = get_mock_properties(entities)
    
    return properties

# ========== RESPONSE GENERATION ==========
def generate_near_landmark_response(entities: Dict[str, Any], properties: List[Dict[str, Any]]) -> str:
    """Generate response for properties near landmarks"""
    
    landmark = entities.get('landmark', 'that location')
    landmark_type = entities.get('landmark_type', 'landmark')
    
    if properties:
        response = f"📍 **Properties Near {landmark.title()}**\n\n"
        
        # Categorize properties by verification method
        verified_by_maps = []
        mentioned_in_desc = []
        other_proximity = []
        
        for prop in properties:
            proximity_info = prop.get('proximity_info', {})
            match_type = proximity_info.get('match_type', '')
            
            if match_type == 'google_maps':
                verified_by_maps.append(prop)
            elif match_type in ['proximity_keyword', 'exact_match', 'school_pattern']:
                mentioned_in_desc.append(prop)
            else:
                other_proximity.append(prop)
        
        total_count = len(verified_by_maps) + len(mentioned_in_desc) + len(other_proximity)
        response += f"I found {total_count} properties near {landmark}:\n\n"
        
        # Show Google Maps verified properties first
        if verified_by_maps:
            response += "**📍 Verified by Distance:**\n"
            for i, prop in enumerate(verified_by_maps[:3]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                proximity = prop.get('proximity_info', {})
                distance = proximity.get('distance', 'Distance not available')
                landmark_name = proximity.get('landmark_found', landmark)
                
                response += f"{i+1}. **{title}** in {location}\n"
                response += f"   💰 {price}\n"
                response += f"   📍 **Distance:** {distance} from {landmark_name}\n"
                
                if proximity.get('duration'):
                    response += f"   🚶 **Walking time:** {proximity['duration']}\n"
                
                response += "\n"
        
        # Show properties mentioned in description
        if mentioned_in_desc:
            response += "**📝 Mentioned in Description:**\n"
            for i, prop in enumerate(mentioned_in_desc[:3]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                proximity = prop.get('proximity_info', {})
                distance = proximity.get('distance', 'Mentioned in description')
                
                response += f"{i+1}. **{title}** in {location} - {price}\n"
                response += f"   ✅ {distance}\n\n"
        
        # Show other proximity properties
        if other_proximity and not (verified_by_maps or mentioned_in_desc):
            response += "**🏫 School-Area Properties:**\n"
            for i, prop in enumerate(other_proximity[:3]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                response += f"{i+1}. **{title}** in {location} - {price}\n"
                
                # Show school-related features if available
                amenities = prop.get('amenities', [])
                school_amenities = [a for a in amenities if any(kw in str(a).lower() 
                                                               for kw in ['school', 'education', 'campus'])]
                if school_amenities:
                    response += f"   ✅ Amenities: {', '.join(school_amenities[:2])}\n"
                
                response += "\n"
        
        # Add tips based on landmark type
        if landmark_type == 'school':
            response += "**🏫 School Proximity Benefits:**\n"
            response += "• **Walking distance** (≤2km) saves transportation time and costs\n"
            response += "• **Educational access** for children's development\n"
            response += "• **Property value** tends to be more stable near good schools\n"
            response += "• **Community** often family-friendly with similar-aged children\n\n"
            
            response += "**🔍 Search Tips:**\n"
            response += "• Try specific school names: *'properties near Batangas State University'*\n"
            response += "• Add property type: *'houses near schools in Lipa'*\n"
            response += "• Specify budget: *'condos near schools under 3M'*\n"
            
        elif landmark_type == 'hospital':
            response += "**🏥 Hospital Proximity Benefits:**\n"
            response += "• **Emergency access** for medical needs\n"
            response += "• **Convenience** for regular medical checkups\n"
            response += "• **Medical professionals** often live nearby\n"
            response += "• **24/7 access** to healthcare services\n"
            
        elif landmark_type == 'mall':
            response += "**🛍️ Mall Proximity Benefits:**\n"
            response += "• **Shopping convenience** within walking distance\n"
            response += "• **Entertainment options** like cinemas and restaurants\n"
            response += "• **Public transportation** hubs are often nearby\n"
            response += "• **Urban lifestyle** with easy access to amenities\n"
        
    else:
        response = f"I couldn't find any properties near {landmark}.\n\n"
        response += "💡 **Suggestions:**\n"
        response += f"• Try a different landmark (schools, hospitals, malls, etc.)\n"
        response += "• Search in a specific location: *'properties near schools in Batangas City'*\n"
        response += "• Broaden your search: *'properties in areas with good schools'*\n"
        response += "• Check properties that mention schools in their descriptions\n"
    
    return response

def generate_documents_only_response(entities: Dict[str, Any]) -> str:
    """Generate response ONLY for document requirements (no properties)"""
    
    financing_type = entities.get('financing_type') or entities.get('sale_type')
    financing_option = entities.get('financing_options')
    
    if financing_type == 'bank_financing' or 'bank' in str(financing_type).lower() or financing_option:
        response = "🏦 **Bank Financing Requirements**\n\n"
        response += "Here are the documents typically needed for bank financing:\n\n"
        
        response += "**📋 Applicant's Requirements:**\n"
        response += "1. **Valid IDs** (any 2 government-issued):\n"
        response += "   • Passport\n"
        response += "   • Driver's License\n"
        response += "   • SSS/GSIS ID\n"
        response += "   • PRC ID\n"
        response += "   • Voter's ID\n\n"
        
        response += "2. **Proof of Income:**\n"
        response += "   • For Employed: 3-6 months payslips\n"
        response += "   • Certificate of Employment with compensation\n"
        response += "   • ITR (Income Tax Return) with BIR stamp\n"
        response += "   • For OFW: Employment contract, payslips\n\n"
        
        response += "3. **Financial Documents:**\n"
        response += "   • Bank Statements (6 months)\n"
        response += "   • Proof of other income sources\n"
        response += "   • List of assets and liabilities\n\n"
        
        response += "**🏦 Property Documents (from Seller):**\n"
        response += "1. **Title Documents:**\n"
        response += "   • Original Certificate of Title (OCT) or Transfer Certificate of Title (TCT)\n"
        response += "   • Tax Declaration\n"
        response += "   • Latest Real Property Tax Receipt\n\n"
        
        response += "2. **Property Documents:**\n"
        response += "   • Location Plan/Vicinity Map\n"
        response += "   • Copy of Deed of Sale\n"
        response += "   • Seller's valid IDs\n\n"
        
        response += "**🏦 Bank-Specific Requirements:**\n"
        if financing_option == 'BDO':
            response += "• **BDO Home Loan Application Form**\n"
            response += "• Credit Report Authorization\n"
            response += "• Property Appraisal Report\n"
            response += "• Contact: (02) 8631-8000 | www.bdo.com.ph\n"
        elif financing_option == 'Metrobank':
            response += "• **Metrobank Housing Loan Application**\n"
            response += "• Disclosure Statement\n"
            response += "• Property Inspection Report\n"
            response += "• Contact: (02) 8888-7000 | www.metrobank.com.ph\n"
        elif 'Pag-IBIG' in str(financing_option):
            response += "**🏦 Pag-IBIG Housing Loan Requirements:**\n"
            response += "• Pag-IBIG Membership ID\n"
            response += "• 24 months contributions (minimum)\n"
            response += "• Housing Loan Application Form\n"
            response += "• Property Appraisal\n"
            response += "• **Loan Amount:** Up to ₱6M\n"
            response += "• **Interest Rate:** As low as 3% per annum\n"
            response += "• **Term:** Up to 30 years\n"
            response += "• **Hotline:** (02) 8724-4244\n"
        else:
            response += "• Completed loan application form\n"
            response += "• Credit report authorization\n"
            response += "• Property appraisal documents\n\n"
        
        response += "**⏱️ Processing Time:** 2-4 weeks after complete document submission\n"
        
    elif financing_type == 'installment':
        response = "📋 **Documents Needed for Installment Purchase**\n\n"
        response += "**For Reservation:**\n"
        response += "1. Reservation Fee (varies by property)\n"
        response += "2. Valid ID (passport, driver's license, etc.)\n\n"
        
        response += "**For Contract Signing:**\n"
        response += "1. 2 Valid IDs (photocopy and original for verification)\n"
        response += "2. Proof of Income (3 months payslips)\n"
        response += "3. Certificate of Employment\n"
        response += "4. Post-dated checks for monthly payments\n"
        response += "5. 2x2 ID pictures (2 copies)\n\n"
        
        response += "**Additional for Self-Employed/Business Owners:**\n"
        response += "• ITR (Income Tax Return) for the last 2 years\n"
        response += "• Business registration (DTI/SEC)\n"
        response += "• Bank statements (6 months)\n"
        response += "• Financial statements\n\n"
        
        response += "**📊 Typical Installment Terms:**\n"
        response += "• Downpayment: 20-30% of property price\n"
        response += "• Payment Term: 3-5 years\n"
        response += "• Interest Rate: 6-8% per annum\n"
        response += "• Monthly amortization\n\n"
        
        response += "**⚖️ Legal Documents:**\n"
        response += "• Contract to Sell\n"
        response += "• Deed of Absolute Sale (upon full payment)\n"
        response += "• Transfer of Title documents\n"
        
    elif financing_type == 'outright' or 'cash' in str(financing_type).lower():
        response = "💰 **Documents for Outright/Cash Purchase**\n\n"
        response += "**Buyer's Requirements:**\n"
        response += "1. Valid IDs (2 government-issued)\n"
        response += "2. Proof of Billing (for address verification)\n"
        response += "3. Proof of Funds/Source of Cash\n"
        response += "   • Bank certification\n"
        response += "   • Bank statements (6 months)\n"
        response += "   • For large amounts: Source of wealth documentation\n\n"
        
        response += "**Property Documents (from Seller):**\n"
        response += "1. Clean Title (no liens/encumbrances)\n"
        response += "2. Tax Declaration\n"
        response += "3. Latest Real Property Tax Receipt\n"
        response += "4. Certificate of No Improvement (if vacant lot)\n"
        response += "5. Location Plan/Vicinity Map\n\n"
        
        response += "**Transaction Documents:**\n"
        response += "• Deed of Absolute Sale\n"
        response += "• Notarization documents\n"
        response += "• Tax clearance (Capital Gains Tax, Documentary Stamp Tax)\n"
        response += "• Transfer of Title at Registry of Deeds\n"
        
    else:
        response = "📋 **General Property Purchase Documents**\n\n"
        response += "**For All Property Transactions:**\n"
        response += "1. **Valid Identification:**\n"
        response += "   • 2 government-issued IDs (passport, driver's license, etc.)\n\n"
        
        response += "2. **Proof of Billing Address:**\n"
        response += "   • Utility bill (electricity, water, telco)\n"
        response += "   • Credit card statement\n\n"
        
        response += "3. **Financial Capacity Proof:**\n"
        response += "   • Bank statements (3-6 months)\n"
        response += "   • Proof of income\n"
        response += "   • ITR (for self-employed)\n\n"
        
        response += "**Additional Based on Payment Method:**\n"
        response += "• **Bank Financing:** Loan application, credit report, property appraisal\n"
        response += "• **Installment:** Post-dated checks, installment agreement\n"
        response += "• **Outright:** Proof of funds, bank certification\n\n"
        
        response += "**💡 Tip:** Requirements may vary by developer, bank, or property type.\n"
        response += "It's best to confirm specific requirements with your chosen financing partner.\n"
    
    return response

def generate_financing_response(entities: Dict[str, Any], properties: List[Dict[str, Any]]) -> str:
    """Generate response for financing-related queries"""
    
    if entities.get('documents_only'):
        return generate_documents_only_response(entities)
    
    sale_type = entities.get('sale_type')
    financing_option = entities.get('financing_options')
    
    if sale_type == 'installment':
        if properties:
            response = f"🏦 **Properties Available for Installment Purchase**\n\n"
            response += f"I found {len(properties)} properties that can be purchased via installment:\n\n"
            
            for i, prop in enumerate(properties[:5]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                response += f"{i+1}. **{title}** in {location}\n"
                response += f"   💰 Sale Price: {price}\n"
                
                installment_details = prop.get('installment_details')
                if installment_details:
                    response += f"   📊 **Installment Estimate:**\n"
                    response += f"      • Downpayment: ₱{installment_details['downpayment']:,.0f} (30%)\n"
                    response += f"      • Monthly: ₱{installment_details['monthly_payment']:,.0f} for 5 years\n"
                    response += f"      • Interest Rate: {installment_details['interest_rate']}\n"
                
                financing_options = prop.get('financingOptions', [])
                if financing_options:
                    response += f"   🏦 **Financing Options:** {', '.join(financing_options[:3])}\n"
                
                response += "\n"
            
            response += "\n**📝 Installment Purchase Process:**\n"
            response += "1. Submit reservation with downpayment (usually 20-30%)\n"
            response += "2. Sign Contract to Sell\n"
            response += "3. Submit required documents\n"
            response += "4. Issue post-dated checks for monthly payments\n"
            response += "5. Receive property title upon full payment\n\n"
            
            response += "**📋 Required Documents:**\n"
            response += "• Valid ID (passport, driver's license)\n"
            response += "• Proof of Income (3 months payslips)\n"
            response += "• Certificate of Employment\n"
            response += "• Post-dated checks\n"
            response += "• 2x2 ID pictures\n"
            
        else:
            response = "❌ **No installment properties found**\n\n"
            response += "💡 **Try these alternatives:**\n"
            response += "• Check properties with **bank financing**\n"
            response += "• Look at **outright cash** properties\n"
            response += "• Ask about **developer in-house financing**\n"
            response += "• Consider **Pag-IBIG housing loans**\n\n"
            response += "You can also try:\n"
            response += "• *'show me properties with bank financing'*\n"
            response += "• *'find houses for outright purchase'*\n"
            response += "• *'properties with Pag-IBIG financing'*\n"
        
        return response
    
    elif sale_type == 'bank_financing':
        if properties:
            response = f"🏦 **Properties with Bank Financing**\n\n"
            response += f"I found {len(properties)} properties that accept bank financing:\n\n"
            
            for i, prop in enumerate(properties[:5]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                response += f"{i+1}. **{title}** in {location}\n"
                response += f"   💰 Sale Price: {price}\n"
                
                financing_options = prop.get('financingOptions', [])
                if financing_options:
                    response += f"   🏦 **Bank Options:** {', '.join(financing_options[:3])}\n"
                
                response += "\n"
            
            response += "\n**🏦 Popular Banks for Property Financing:**\n"
            response += "• **BDO** - (02) 8631-8000 | www.bdo.com.ph\n"
            response += "• **Metrobank** - (02) 8888-7000 | www.metrobank.com.ph\n"
            response += "• **UnionBank** - (02) 8841-8600 | www.unionbankph.com\n"
            response += "• **RCBC** - (02) 8557-9515 | www.rcbc.com\n\n"
            
            response += "**📋 Common Requirements for Bank Financing:**\n"
            response += "1. Valid ID\n"
            response += "2. Proof of Income (3-6 months)\n"
            response += "3. Certificate of Employment\n"
            response += "4. ITR (Income Tax Return)\n"
            response += "5. Bank Statements\n"
            response += "6. Property Documents\n"
            
        else:
            response = "❌ **No properties found with bank financing**\n\n"
            response += "💡 **Try searching for sale properties first:**\n"
            response += "• *'find houses for sale in Batangas City'*\n"
            response += "• *'show me condos for sale'*\n"
            response += "• *'properties for sale with financing options'*\n"
        
        return response
    
    elif sale_type == 'outright':
        if properties:
            response = f"💰 **Properties for Outright Purchase (Cash)**\n\n"
            response += f"I found {len(properties)} properties available for outright cash purchase:\n\n"
            
            for i, prop in enumerate(properties[:5]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                response += f"{i+1}. **{title}** in {location}\n"
                response += f"   💰 Sale Price: {price}\n"
                response += f"   📋 Payment: Cash/Outright\n\n"
            
            response += "\n**💰 Benefits of Outright Purchase:**\n"
            response += "• No interest payments\n"
            response += "• Faster transaction process\n"
            response += "• Potential for price negotiation\n"
            response += "• Immediate property transfer\n"
            
        else:
            response = "❌ **No properties found for outright purchase**\n\n"
            response += "💡 **Most properties accept multiple payment options. Try:**\n"
            response += "• *'show me properties for sale'*\n"
            response += "• *'find houses with different payment options'*\n"
            response += "• *'what payment methods do you accept'*\n"
        
        return response
    
    elif financing_option:
        if properties:
            response = f"🏦 **Properties with {financing_option} Financing**\n\n"
            response += f"I found {len(properties)} properties that accept {financing_option}:\n\n"
            
            for i, prop in enumerate(properties[:5]):
                title = prop.get('title', f'Property {i+1}')
                price = prop.get('price', 'Price not available')
                location = prop.get('location', 'Location not specified')
                
                response += f"{i+1}. **{title}** in {location}\n"
                response += f"   💰 Price: {price}\n"
                
                financing_options = prop.get('financingOptions', [])
                if financing_options:
                    response += f"   🏦 **Available Options:** {', '.join(financing_options[:3])}\n"
                
                response += "\n"
            
            if 'BDO' in financing_option:
                response += "\n**🏦 BDO Home Loan Features:**\n"
                response += "• Loan Amount: Up to 80% of property value\n"
                response += "• Term: Up to 25 years\n"
                response += "• Interest Rate: Competitive rates\n"
                response += "• Contact: (02) 8631-8000\n"
                
            elif 'Pag-IBIG' in financing_option:
                response += "\n**🏦 Pag-IBIG Housing Loan:**\n"
                response += "• Membership: At least 24 months\n"
                response += "• Maximum Loan: ₱6M\n"
                response += "• Term: Up to 30 years\n"
                response += "• Interest: As low as 3% per annum\n"
                response += "• Hotline: (02) 8724-4244\n"
            
        else:
            response = f"❌ **No properties found with {financing_option} financing**\n\n"
            response += "💡 **Try these suggestions:**\n"
            response += f"• Ask about other banks or financing options\n"
            response += "• Check if properties accept multiple financing options\n"
            response += "• Look for sale properties and inquire about financing\n"
        
        return response
    
    else:
        response = "🏦 **Financing Options for Property Purchase**\n\n"
        response += "We offer various financing options for property purchases:\n\n"
        response += "**1. Installment Plans**\n"
        response += "   • Developer in-house financing\n"
        response += "   • Flexible payment terms\n"
        response += "   • Usually 20-30% downpayment\n\n"
        
        response += "**2. Bank Financing**\n"
        response += "   • **BDO** - (02) 8631-8000\n"
        response += "   • **Metrobank** - (02) 8888-7000\n"
        response += "   • **UnionBank** - (02) 8841-8600\n"
        response += "   • **RCBC** - (02) 8557-9515\n\n"
        
        response += "**3. Pag-IBIG Housing Loan**\n"
        response += "   • For members with 24+ months contributions\n"
        response += "   • Up to ₱6M loan amount\n"
        response += "   • As low as 3% interest\n"
        response += "   • Hotline: (02) 8724-4244\n\n"
        
        response += "**4. Outright/Cash Purchase**\n"
        response += "   • No interest payments\n"
        response += "   • Faster transaction\n"
        response += "   • Potential discounts\n\n"
        
        response += "💡 **To see properties with specific financing, try:**\n"
        response += "• *'show me properties that accept installment'*\n"
        response += "• *'find houses with bank financing'*\n"
        response += "• *'properties with Pag-IBIG loan'*\n"
        response += "• *'outright purchase properties'*\n"
        
        return response

def generate_criteria_search_response(entities: Dict[str, Any], properties: List[Dict[str, Any]]) -> str:
    """Generate response for property searches with specific criteria"""
    
    filtered_properties = []
    for prop in properties:
        matches = True
        
        if entities.get('max_price'):
            price_numeric = prop.get('price_numeric', 0)
            if price_numeric > entities['max_price']:
                matches = False
        
        if entities.get('exact_bedrooms') is not None:
            prop_bedrooms = prop.get('bedrooms', 'Not specified')
            try:
                if isinstance(prop_bedrooms, str):
                    bed_match = re.search(r'(\d+)', str(prop_bedrooms))
                    if bed_match:
                        prop_bed_num = int(bed_match.group(1))
                    else:
                        prop_bed_num = 0
                else:
                    prop_bed_num = int(prop_bedrooms)
                
                if prop_bed_num != entities['exact_bedrooms']:
                    matches = False
            except:
                pass
        
        if matches:
            filtered_properties.append(prop)
    
    properties = filtered_properties
    
    criteria_parts = []
    
    if entities.get('property_type'):
        prop_type = entities['property_type'].replace('_', ' ').title()
        criteria_parts.append(f"{prop_type}")
    else:
        criteria_parts.append("properties")
    
    if entities.get('exact_bedrooms') is not None:
        bedrooms = entities['exact_bedrooms']
        criteria_parts.append(f"with {bedrooms} bedroom{'s' if bedrooms != 1 else ''}")
    
    if entities.get('max_price'):
        max_price = entities['max_price']
        if max_price >= 1000000:
            criteria_parts.append(f"under ₱{max_price/1000000:.1f}M")
        else:
            criteria_parts.append(f"under ₱{max_price:,.0f}")
    
    if entities.get('location'):
        criteria_parts.append(f"in {entities['location']}")
    
    criteria_desc = " ".join(criteria_parts)
    
    if properties:
        properties_by_location = {}
        for prop in properties:
            location = prop.get('city', 'Unknown')
            if location not in properties_by_location:
                properties_by_location[location] = []
            properties_by_location[location].append(prop)
        
        response = f"🔍 **Found {len(properties)} {criteria_desc}**\n\n"
        
        for location, loc_props in properties_by_location.items():
            response += f"📍 **{location}** ({len(loc_props)} available)\n"
            
            for prop in loc_props[:3]:
                title = prop.get('title', 'Property')
                price = prop.get('price', 'Price not available')
                prop_type = prop.get('type', '').replace('_', ' ')
                
                prop_bedrooms = prop.get('bedrooms', '')
                if prop_bedrooms:
                    bed_display = f" | 🛏️ {prop_bedrooms}"
                else:
                    bed_display = ""
                
                response += f"   • **{title}** ({prop_type}) - {price}{bed_display}\n"
            
            response += "\n"
        
        if len(properties) > 10:
            response += f"*Showing {min(len(properties), 10)} of {len(properties)} properties.*\n\n"
        
        if len(properties) < 3:
            response += "💡 **Tips for more results:**\n"
            response += "   • Expand your price range\n"
            response += "   • Consider nearby locations\n"
            if entities.get('exact_bedrooms'):
                response += "   • Try different bedroom counts\n"
        
    else:
        response = f"I found 0 {criteria_desc}.\n\n"
        response += "💡 **Suggestions:**\n"
        response += "• Try a different price range\n"
        response += "• Consider nearby locations\n"
        if entities.get('exact_bedrooms'):
            response += "• Adjust your bedroom requirements\n"
        response += "• Check back later for new listings\n"
    
    return response

def generate_general_search_response(entities: Dict[str, Any], properties: List[Dict[str, Any]]) -> str:
    """Generate response for general property searches without location"""
    
    property_type = entities.get('property_type', 'properties')
    property_type_display = property_type.replace('_', ' ').title()
    exact_bedrooms = entities.get('exact_bedrooms')
    
    if properties:
        properties_by_city = defaultdict(list)
        for prop in properties:
            city = prop.get('city', 'Unknown City')
            properties_by_city[city].append(prop)
        
        sorted_cities = sorted(properties_by_city.items(), key=lambda x: len(x[1]), reverse=True)
        
        response = f"🔍 **{property_type_display} Available in Batangas**\n\n"
        
        criteria_parts = [property_type_display.lower()]
        if exact_bedrooms is not None:
            criteria_parts.append(f"with {exact_bedrooms} bedroom{'s' if exact_bedrooms != 1 else ''}")
        
        criteria_desc = " ".join(criteria_parts)
        response += f"I found {len(properties)} {criteria_desc} across different locations:\n\n"
        
        displayed_count = 0
        for city, city_props in sorted_cities[:5]:
            if displayed_count >= 15:
                break
                
            response += f"**📍 {city}** ({len(city_props)} available)\n"
            
            for i, prop in enumerate(city_props[:3]):
                title = prop.get('title', f'{property_type_display} {i+1}')
                price = prop.get('price', 'Price not available')
                prop_type = prop.get('type', property_type).replace('_', ' ')
                
                response += f"   • **{title}** ({prop_type}) - {price}\n"
                displayed_count += 1
            
            response += "\n"
        
        if len(properties) > displayed_count:
            response += f"\n*Showing {displayed_count} of {len(properties)} {property_type_display.lower()}. "
            response += f"Properties found in {len(properties_by_city)} different locations.*\n"
        else:
            response += f"\n*Properties found in {len(properties_by_city)} different locations.*\n"
        
        response += "\n💡 **Tips for better results:**\n"
        response += "   • Add a location: *'find apartments in Batangas City'*\n"
        response += "   • Specify budget: *'find houses under 3M'*\n"
        response += "   • Add features: *'find condos with swimming pool'*\n"
        response += "   • Specify needs: *'find properties for family'*\n"
        
        if property_type in ['house', 'condo', 'apartment']:
            response += "\n📍 **Popular locations for " + property_type_display.lower() + ":**\n"
            response += "   • Batangas City (urban living, near port)\n"
            response += "   • Lipa City (cool climate, educational hub)\n"
            response += "   • Nasugbu (beachfront, vacation homes)\n"
            response += "   • Sto. Tomas City (near Metro Manila)\n"
            response += "   • Tanauan City (Taal Lake views)\n"
        
    else:
        response = f"I found 0 {property_type_display.lower()}"
        
        if exact_bedrooms is not None:
            response += f" with {exact_bedrooms} bedroom{'s' if exact_bedrooms != 1 else ''}"
        
        response += ".\n\n"
        
        response += "💡 **Suggestions:**\n"
        response += "• Try a different location\n"
        response += "• Adjust your criteria\n"
        response += "• Check back later for new listings\n"
    
    return response

def generate_response(intent: str, entities: Dict[str, Any], properties: List[Dict[str, Any]]) -> str:
    """Generate response based on intent and entities using training data templates"""
    
    if intent == 'find_property_for_need':
        # Special handling for family needs
        if entities.get('family_info'):
            family_type = entities['family_info']
            if isinstance(family_type, dict):
                family_size = family_type.get('value', 3)
            else:
                family_size = 3  # Default
            
            # Use specialized family response generator
            return generate_family_needs_response(family_size, properties, entities)
    
    if intent == 'financing':
        return generate_financing_response(entities, properties)
    
    if intent == 'find_near_landmark':
        return generate_near_landmark_response(entities, properties)
    
    if intent == 'find_property_with_criteria':
        return generate_criteria_search_response(entities, properties)
    
    if intent == 'find_property' and entities.get('has_general_search'):
        return generate_general_search_response(entities, properties)
       
    default_responses = {
        'find_property': "I understand you're looking for properties. Could you specify the location or property type?",
        'find_near_landmark': "I can help you find properties near landmarks. What specific landmark are you interested in?",
        'financing': "I can provide information about financing options. Which type of financing are you interested in?",
        'location_info': "I can tell you about different locations in Batangas. Which location would you like to know about?",
        'find_with_feature': "I can help you find properties with specific features. What feature are you looking for?",
        'find_ready_property': "I can help you find ready-to-move-in properties. What location are you interested in?",
        'process_info': "I can explain property purchase processes. What specific process are you interested in?",
        'match_needs': "I can match properties to your needs. What are your specific requirements?",
        'find_property_for_need': "I can find properties suitable for specific needs. What type of need are you looking for?",
        'find_property_with_criteria': "I can find properties matching specific criteria. What criteria do you have?",
        'unknown': "I understand you're looking for property information in Batangas. Could you provide more details about what you need?"
    }
    
    if training_data and 'training_samples' in training_data:
        matching_samples = [s for s in training_data['training_samples'] if s.get('intent') == intent]
        
        if matching_samples:
            best_sample = None
            for sample in matching_samples:
                sample_entities = sample.get('entities', {})
                
                match_score = 0
                for key, value in sample_entities.items():
                    if entities.get(key) and value and str(value).lower() in str(entities.get(key)).lower():
                        match_score += 1
                
                if match_score > 0 and (not best_sample or match_score > best_sample.get('match_score', 0)):
                    sample['match_score'] = match_score
                    best_sample = sample
            
            if best_sample and 'response_template' in best_sample:
                template = best_sample['response_template']
                
                replacements = {
                    '{count}': str(len(properties)),
                    '{property_type}': entities.get('property_type', 'property'),
                    '{location}': entities.get('location', 'the area'),
                    '{financing_type}': entities.get('financing_type', 'financing'),
                    '{feature}': entities.get('feature', 'feature'),
                    '{landmark}': entities.get('landmark', 'landmark'),
                    '{bedrooms}': str(entities.get('bedrooms', '')),
                    '{price_range}': entities.get('price_range', '')
                }
                
                if properties:
                    property_list = "\n"
                    for i, prop in enumerate(properties[:3]):
                        title = prop.get('title', f'Property {i+1}')
                        price = prop.get('price', 'Price not available')
                        location = prop.get('location', 'Location not specified')
                        property_list += f"{i+1}. **{title}** in {location} - {price}\n"
                    replacements['{property_list}'] = property_list
                else:
                    replacements['{property_list}'] = "No specific properties found with those criteria."
                
                for key, value in best_sample.items():
                    if key.startswith('location_description') or key.startswith('average_') or key in ['documents_list', 'requirements_list', 'key_features', 'average_prices', 'ideal_for', 'property_types']:
                        if value is not None:
                            if isinstance(value, list):
                                replacements[f'{{{key}}}'] = '\n'.join([f"• {item}" for item in value])
                            else:
                                replacements[f'{{{key}}}'] = str(value)
                
                response = template
                for placeholder, replacement in replacements.items():
                    if replacement is None:
                        replacement = ''
                    response = response.replace(placeholder, str(replacement))
                
                if intent == 'location_info' and entities.get('location'):
                    location_name = entities['location']
                    if training_data and 'location_profiles' in training_data:
                        location_profile = training_data['location_profiles'].get(location_name)
                        if location_profile:
                            for key, value in location_profile.items():
                                if value is not None:
                                    response = response.replace(f'{{{key}}}', str(value))
                
                return response
    
    response = default_responses.get(intent, default_responses['unknown'])
    
    if intent == 'location_info' and entities.get('location'):
        location_name = entities['location']
        if training_data and 'location_profiles' in training_data:
            location_profile = training_data['location_profiles'].get(location_name)
            if location_profile:
                description = location_profile.get('description', 'No description available.')
                lifestyle = location_profile.get('lifestyle', 'No lifestyle information available.')
                
                response = f"📍 **About {location_name}**\n"
                response += f"**Description:** {description}\n\n"
                response += f"**Lifestyle:** {lifestyle}\n\n"
                
                if 'key_features' in location_profile and location_profile['key_features']:
                    response += "**Key Features:**\n"
                    for feature in location_profile['key_features']:
                        response += f"• {feature}\n"
                    response += "\n"
                
                if 'average_prices' in location_profile and location_profile['average_prices']:
                    response += "**Average Property Prices:**\n"
                    for price_info in location_profile['average_prices']:
                        response += f"• {price_info}\n"
                    response += "\n"
                
                if 'ideal_for' in location_profile and location_profile['ideal_for']:
                    response += f"**Ideal For:** {', '.join(location_profile['ideal_for'])}\n\n"
                
                if 'property_types' in location_profile and location_profile['property_types']:
                    response += f"**Property Types Available:** {', '.join(location_profile['property_types'])}\n"
                
                if properties and len(properties) > 0:
                    response += "\n**Available Properties:**\n"
                    for i, prop in enumerate(properties[:3]):
                        title = prop.get('title', f'Property {i+1}')
                        price = prop.get('price', 'Price not available')
                        location = prop.get('location', 'Location not specified')
                        response += f"{i+1}. **{title}** in {location} - {price}\n"
                
                return response
        else:
            response = f"I can tell you about {location_name} in Batangas.\n\n"
            response += f"{location_name} is one of the key locations in Batangas province with various property options available.\n\n"
            response += "If you're interested in properties here, you might want to specify what type of property you're looking for (apartment, house, condo, etc.) or your budget range."
    
    elif properties and len(properties) > 0:
        response += "\n\n**Available Properties:**\n"
        for i, prop in enumerate(properties[:3]):
            title = prop.get('title', f'Property {i+1}')
            price = prop.get('price', 'Price not available')
            location = prop.get('location', 'Location not specified')
            response += f"{i+1}. **{title}** in {location} - {price}\n"
    
    return response

def determine_intent_fallback(query: str) -> str:
    """Simple rule-based intent detection as fallback"""
    query_lower = query.lower()

        # Family/needs-based queries
    needs_keywords = [
        'for family', 'for families', 'for couple', 'for couples',
        'for students', 'for professionals', 'for retirees',
        'for business', 'for investors', 'for single', 'for workers'
    ]
    
    for keyword in needs_keywords:
        if keyword in query_lower:
            return 'find_property_for_need'
    
    doc_keywords = ['documents', 'requirements', 'needed', 'required', 'paperwork']
    prop_keywords = ['properties', 'show me', 'find', 'looking for', 'search']
    
    has_doc_keywords = any(term in query_lower for term in doc_keywords)
    has_prop_keywords = any(term in query_lower for term in prop_keywords)
    
    if has_doc_keywords and not has_prop_keywords:
        return 'financing'
    
    financing_keywords = [
        'installment', 'bank financing', 'mortgage', 'loan',
        'financing', 'payment plan', 'pag-ibig', 'bdo',
        'metrobank', 'unionbank', 'rcbc', 'housing loan',
        'accept bank', 'accept installment', 'outright', 'cash'
    ]
    
    for keyword in financing_keywords:
        if keyword in query_lower:
            return 'financing'
    
    if any(term in query_lower for term in ['find', 'search', 'looking for', 'show me', 'need', 'want']):
        return 'find_property'
    
    if any(term in query_lower for term in ['tell me about', 'information about', 'describe', 'about']):
        return 'location_info'
    
    return 'unknown'

# ========== API ENDPOINTS ==========
@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chatbot endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"💬 Query: '{query}'")
        
        # Convert to lowercase once for use throughout
        query_lower = query.lower()
        
        # Step 1: Predict intent
        intent = "unknown"
        confidence = 0.0
        
        if vectorizer and classifier:
            try:
                processed_query = preprocess_text(query)
                X = vectorizer.transform([processed_query])
                intent = classifier.predict(X)[0]
                proba = classifier.predict_proba(X)[0]
                confidence = float(max(proba))
                logger.info(f"🎯 Intent: {intent} (confidence: {confidence:.2%})")
                
                if confidence < 0.7:
                    top_indices = np.argsort(proba)[-3:][::-1]
                    logger.info("   Low confidence alternatives:")
                    for idx in top_indices:
                        alt_intent = model_classes[idx] if idx < len(model_classes) else "unknown"
                        alt_prob = proba[idx]
                        logger.info(f"     • {alt_intent}: {alt_prob:.2%}")
                        
            except Exception as e:
                logger.error(f"❌ Model prediction failed: {e}")
                intent = determine_intent_fallback(query)
        else:
            intent = determine_intent_fallback(query)
            # query_lower is already defined above
        
        # Define patterns that should be find_property_for_need
        family_need_patterns = [
            'properties for couple',
            'properties for couples',
            'for couple',
            'for couples',
            'for family',
            'for families',
            'family of',
            'family with',
            'for students',
            'for professionals'
        ]
        
        # Check if query matches family need patterns
        is_family_need_query = any(pattern in query_lower for pattern in family_need_patterns)
        
        if is_family_need_query:
            # Override intent to find_property_for_need
            old_intent = intent
            intent = 'find_property_for_need'
            confidence = max(confidence, 0.9)  # Ensure high confidence
            logger.info(f"🔄 Overriding intent from {old_intent} to {intent} for family/need query: '{query}'")
        
        # Special handling for "near schools" queries
        if 'near school' in query_lower or 'near schools' in query_lower:
            if intent != 'find_near_landmark':
                old_intent = intent
                intent = 'find_near_landmark'
                confidence = max(confidence, 0.9)
                logger.info(f"🔄 Overriding intent from {old_intent} to {intent} for school proximity query")
                
        # Step 2: Extract entities
        entities = extract_entities_from_query(query)
        logger.info(f"🏷️ Entities: {entities}")
        
        # Step 3: Search properties if needed
        properties = []
        if intent in ["find_property", "find_near_landmark", "find_with_feature", 
                     "find_ready_property", "find_property_for_need", 
                     "find_property_with_criteria", "match_needs", "financing"]:
            properties = search_firestore_properties(entities)

        if 'couple' in query_lower or 'couples' in query_lower:
            if not entities.get('family_info'):
                entities['family_info'] = {'type': 'couple'}
                entities['has_need_query'] = True
                entities['min_bedrooms'] = 1
                entities['ideal_bedrooms'] = 2
                logger.info(f"👨‍👩‍👧‍👦 Added missing family info for couple query")
        
        # Step 4: Generate response
        if entities.get('documents_only'):
            response_text = generate_documents_only_response(entities)
        else:
            response_text = generate_response(intent, entities, properties)
        
        # Step 5: Prepare result
        result = {
            'success': True,
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'response': response_text,
            'properties_found': len(properties),
            'properties': properties[:10] if not entities.get('documents_only') else [],
            'model_version': 'trained' if vectorizer else 'fallback',
            'is_general_search': entities.get('has_general_search', False),
            'is_criteria_search': intent == 'find_property_with_criteria',
            'is_financing_query': intent == 'financing',
            'is_document_query': entities.get('documents_only', False),
            'is_landmark_query': intent == 'find_near_landmark',
            'landmark_type': entities.get('landmark_type'),
            'google_maps_available': GOOGLE_MAPS_AVAILABLE
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I encountered an error processing your request. Please try again with a different query."
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Bah.AI Property Chatbot',
        'version': '3.7.0',
        'model_loaded': vectorizer is not None and classifier is not None,
        'training_data_loaded': bool(training_data),
        'firebase_connected': db is not None,
        'google_maps_available': GOOGLE_MAPS_AVAILABLE,
        'model_intents': model_classes,
        'model_features': len(vectorizer.get_feature_names_out()) if vectorizer else 0,
        'spacy_loaded': False, 
        'supports_general_searches': True,
        'supports_criteria_searches': True,
        'supports_financing_queries': True,
        'supports_document_queries': True,
        'supports_landmark_queries': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify the model is working"""
    test_queries = [
        "properties near schools",
        "houses near batangas state university",
        "apartments near colleges",
        "properties close to schools",
        "condos walking distance to campus",
        "family homes near good schools",
        "student housing near universities",
        "requirements for bank financing",
        "what documents are needed for installment",
        "documents required for outright purchase",
        "show me properties that accept installment",
        "find properties with bank financing",
        "properties that accept outright payment",
        "houses with BDO financing",
        "condos with Pag-IBIG loan",
        "show me houses under 15M with 3 bedrooms",
        "find condos below 10M with 2 bedrooms",
        "find apartments",
        "show me houses",
        "find apartments in batangas city",
    ]
    
    results = []
    for query in test_queries:
        try:
            if vectorizer and classifier:
                processed = preprocess_text(query)
                X = vectorizer.transform([processed])
                intent = classifier.predict(X)[0]
                confidence = float(classifier.predict_proba(X).max())
                
                entities = extract_entities_from_query(query)
                
                results.append({
                    'query': query,
                    'intent': intent,
                    'confidence': confidence,
                    'landmark': entities.get('landmark'),
                    'landmark_type': entities.get('landmark_type'),
                    'documents_only': entities.get('documents_only'),
                    'sale_type': entities.get('sale_type'),
                    'financing_options': entities.get('financing_options'),
                    'has_location': entities.get('location') is not None,
                    'property_type': entities.get('property_type'),
                    'max_price': entities.get('max_price'),
                    'exact_bedrooms': entities.get('exact_bedrooms'),
                })
        except Exception as e:
            results.append({
                'query': query,
                'error': str(e)
            })
    
    return jsonify({
        'test_results': results,
        'model_status': 'loaded' if vectorizer else 'not loaded',
        'training_data_status': 'loaded' if training_data else 'not loaded',
        'google_maps_available': GOOGLE_MAPS_AVAILABLE,
        'supports_criteria_searches': True,
        'supports_general_searches': True,
        'supports_financing_queries': True,
        'supports_document_queries': True,
        'supports_landmark_queries': True
    })

# ========== MAIN APPLICATION ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BAH.AI PROPERTY CHATBOT BACKEND v3.7.0")
    print("   (With Google Maps integration for landmark proximity)")
    print("="*60)
    
    # Load the trained model
    load_nlu_model()
    
    # Load training data for response templates
    load_training_data()
    
    print(f"\n📂 NLU Model: {'✅ Loaded' if vectorizer else '❌ Not loaded'}")
    print(f"📚 Training Data: {'✅ Loaded' if training_data else '❌ Not loaded'}")
    print(f"🔥 Firebase: {'✅ Connected' if db else '❌ Not connected'}")
    print(f"🗺️ Google Maps: {'✅ Available' if GOOGLE_MAPS_AVAILABLE else '❌ Not installed (pip install googlemaps)'}")
    
    if vectorizer:
        print(f"📊 Model intents: {len(model_classes)} intents")
        print(f"📊 Available intents: {', '.join(model_classes)}")
    else:
        print("\n⚠️  WARNING: NLU model not loaded!")
        print("💡 To fix this:")
        print("   1. Run: python train_nlu.py")
        print("   2. Make sure models/nlu_model.pkl exists")
        print("   3. Check the model file path")
    
    print("\n🌐 API Endpoints:")
    print("   GET  /           - Service status")
    print("   POST /api/chat   - Chatbot endpoint")
    print("   GET  /api/health - Health check")
    print("   GET  /api/test   - Test model predictions")
    
    print("\n🔍 Example queries to test:")
    print("   1. 'properties near schools'")
    print("   2. 'houses near batangas state university'")
    print("   3. 'apartments walking distance to campus'")
    print("   4. 'requirements for bank financing'")
    print("   5. 'show me houses under 15M with 3 bedrooms'")
    
    print("="*60 + "\n")
    
    # Get port from environment variable for Render
    port = int(os.environ.get('PORT', 10000))
    print(f"📡 Server would run on port: {port}")
    print("📡 Gunicorn will start the server in production")
    
    # Check for Google Maps API key
    google_maps_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if GOOGLE_MAPS_AVAILABLE and not google_maps_key:
        print("⚠️  WARNING: GOOGLE_MAPS_API_KEY environment variable not set!")
        print("💡 Set it to enable distance calculations for landmark queries")
    elif GOOGLE_MAPS_AVAILABLE and google_maps_key:
        print("✅ Google Maps API key found")
    
    app.run(host='0.0.0.0', port=port, debug=False)