# recommender/app.py - CLEAN WORKING VERSION
import os
import time
from datetime import datetime, timedelta, timezone
import math
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import MinMaxScaler

# Initialize Firebase Admin
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_path:
    raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
# Fix CORS - allow all origins for development
CORS(app, origins="*", supports_credentials=True)

# ---
# CONFIG: Hyperparameters
# ---
SIM_WEIGHT = 0.6
PRICE_WEIGHT = 0.2
RECENCY_WEIGHT = 0.1
POPULARITY_WEIGHT = 0.1
RECENT_WINDOW_DAYS = 30
MAX_DISTANCE_KM = 20.0

# Event weights for user profile building
EVENT_WEIGHTS = {'view': 1.0, 'save': 2.5, 'contact': 3.0, 'inquiry': 3.0}

# ---
# Utils
# ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---
# Feature building
# ---
def build_property_vector(prop, type_vocab, amenity_vocab):
    """Build feature vector for a property"""
    # Get price - handle both rental and sale properties
    price = prop.get('monthlyRent', 0) or prop.get('pricing', 0)
    
    # Numeric features
    bedrooms = prop.get('bedrooms', 0)
    bathrooms = prop.get('bathrooms', 0)
    lat = prop.get('latitude', 0.0)
    lon = prop.get('longitude', 0.0)
    
    # For rent properties, include deposit and minStay
    deposit = prop.get('deposit', 0)
    min_stay = prop.get('minStay', 0)
    
    # Property type one-hot encoding
    prop_type = prop.get('propertyType', '').lower()
    type_vec = [1.0 if t == prop_type else 0.0 for t in type_vocab]
    
    # Amenities multi-hot encoding
    amenities = prop.get('amenities', [])
    amen_vec = [1.0 if a in amenities else 0.0 for a in amenity_vocab]
    
    # Assemble base vector
    base = [price, bedrooms, bathrooms, lat, lon, deposit, min_stay] + type_vec + amen_vec
    return np.array(base, dtype=float)

def normalize_price_vector(matrix):
    """Normalize the price column (first column) using min-max scaling"""
    if matrix.shape[0] == 0:
        return matrix, None
    
    # Min-max normalize price column
    prices = matrix[:, 0].reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices).flatten()
    matrix[:, 0] = scaled
    return matrix, scaler

# ---
# Indexing all properties
# ---
cached_index = {
    "timestamp": None,
    "props": [],
    "ids": [],
    "vectors": None,
    "type_vocab": [],
    "amenity_vocab": [],
    "price_scaler": None
}

def build_index(force=False):
    """Build or retrieve cached property index"""
    global cached_index
    
    # Use cache if recent (10 minutes)
    if cached_index["timestamp"] and not force:
        if time.time() - cached_index["timestamp"] < 600:
            return cached_index
    
    props = []
    ids = []
    type_vocab = set()
    amenity_vocab = set()
    
    # Fetch all active properties
    docs = db.collection('properties').where('status', '==', 'active').stream()
    
    for d in docs:
        p = d.to_dict()
        p['id'] = d.id
        props.append(p)
        ids.append(d.id)
        
        # Build vocabularies
        if p.get('propertyType'):
            type_vocab.add(p['propertyType'].lower())
        
        amenities = p.get('amenities', [])
        for a in amenities:
            amenity_vocab.add(a)
    
    type_vocab = sorted(list(type_vocab))
    amenity_vocab = sorted(list(amenity_vocab))
    
    # Build feature vectors
    vectors = []
    for p in props:
        vec = build_property_vector(p, type_vocab, amenity_vocab)
        vectors.append(vec)
    
    if len(vectors) == 0:
        cached_index.update(
            timestamp=time.time(),
            props=props,
            ids=ids,
            vectors=np.array([]),
            type_vocab=type_vocab,
            amenity_vocab=amenity_vocab,
            price_scaler=None
        )
        return cached_index
    
    # Stack and normalize
    matrix = np.vstack(vectors)
    matrix, scaler = normalize_price_vector(matrix)
    
    cached_index.update(
        timestamp=time.time(),
        props=props,
        ids=ids,
        vectors=matrix,
        type_vocab=type_vocab,
        amenity_vocab=amenity_vocab,
        price_scaler=scaler
    )
    
    return cached_index

# ---
# User profile building
# ---
def get_user_profile_vector(user_id, index):
    """Build user profile vector from preferences and interaction history"""
    if index['vectors'].size == 0:
        return None
    
    M = index['vectors'].shape[1]
    profile = np.zeros(M, dtype=float)
    weight_sum = 0.0
    
    # Fetch user preferences
    user_doc = db.collection('users').document(user_id).get()
    user_pref = user_doc.to_dict() if user_doc.exists else {}
    
    # Get recent interactions - FIXED datetime usage
    since = datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)
    
    # Query events with timestamp filter
    try:
        events_q = db.collection('events') \
            .where('userId', '==', user_id) \
            .where('timestamp', '>=', since) \
            .stream()
    except Exception as e:
        # If index not ready, get all events for user
        print(f"Warning: Using fallback query (index might be building): {e}")
        events_q = db.collection('events') \
            .where('userId', '==', user_id) \
            .stream()
    
    # Aggregate interactions by property
    agg = {}
    for e in events_q:
        ed = e.to_dict()
        pid = ed.get('propertyId')
        etype = ed.get('eventType', 'view')
        w = EVENT_WEIGHTS.get(etype, 1.0)
        agg[pid] = agg.get(pid, 0.0) + w
    
    # Build profile from interactions
    for pid, w in agg.items():
        try:
            idx = index['ids'].index(pid)
        except ValueError:
            continue
        vec = index['vectors'][idx]
        profile += vec * w
        weight_sum += w
    
    # Normalize profile
    if weight_sum > 0:
        profile = profile / weight_sum
    
    return profile

# ---
# Scoring function
# ---
def score_property_for_user(user_vector, prop_vec, prop_meta, user_meta=None):
    """Score a property for a user based on multiple factors"""
    # Similarity score
    sim = cosine_sim(user_vector, prop_vec)
    
    # Price match score
    price_score = 0.5
    if user_meta and user_meta.get('preferredBudget'):
        min_b = user_meta['preferredBudget'].get('min', 0)
        max_b = user_meta['preferredBudget'].get('max', 0)
        raw_price = prop_meta.get('monthlyRent') or prop_meta.get('pricing', 0)
        
        if max_b > 0:
            mid = (min_b + max_b) / 2.0
            diff = abs(raw_price - mid)
            price_score = 1.0 - min(diff / max_b, 1.0)
    
    # Recency boost
    recency_score = 0.0
    if prop_meta.get('createdAt'):
        created_at = prop_meta['createdAt']
        try:
            # Correct way to get Firestore timestamp as datetime
            if hasattr(created_at, 'timestamp'):
                created_dt = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
            elif hasattr(created_at, 'seconds'):
                created_dt = datetime.fromtimestamp(created_at.seconds, tz=timezone.utc)
            else:
                created_dt = created_at  # Assume it's already a datetime
            
            if isinstance(created_dt, (int, float)):
                created_dt = datetime.fromtimestamp(created_dt, tz=timezone.utc)
            
            days_old = (datetime.now(timezone.utc) - created_dt).days
            recency_score = max(0.0, 1.0 - (days_old / RECENT_WINDOW_DAYS))
        except Exception as e:
            app.logger.error(f"Error parsing timestamp: {e}")
            recency_score = 0.0
    
    # Popularity score
    popularity = 0.0
    try:
        ev_q = db.collection('events') \
            .where('propertyId', '==', prop_meta.get('id')) \
            .limit(50) \
            .stream()
        pop_count = sum(1 for _ in ev_q)
        popularity = min(pop_count / 10.0, 1.0)
    except Exception:
        popularity = 0.0
    
    # Combined final score
    final = (SIM_WEIGHT * sim) + \
            (PRICE_WEIGHT * price_score) + \
            (RECENCY_WEIGHT * recency_score) + \
            (POPULARITY_WEIGHT * popularity)
    
    return {
        "similarity": sim,
        "price_score": price_score,
        "recency_score": recency_score,
        "popularity": popularity,
        "final_score": final
    }

# ---
# API endpoints
# ---
@app.route('/recommendations', methods=['GET', 'OPTIONS'])
def recommendations():
    """Get personalized property recommendations for a user"""
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        return '', 200
    
    user_id = request.args.get('userId')
    count = int(request.args.get('count', 10))
    
    if not user_id:
        return jsonify({"error": "userId required"}), 400
    
    # Build property index
    index = build_index()
    
    if index['vectors'].size == 0:
        return jsonify({"results": []})
    
    # Build user profile
    user_vec = get_user_profile_vector(user_id, index)
    
    # If no user vector (no interactions), use zero vector
    if user_vec is None:
        user_vec = np.zeros(index['vectors'].shape[1], dtype=float)
    
    # Get user metadata
    user_meta_doc = db.collection('users').document(user_id).get()
    user_meta = user_meta_doc.to_dict() if user_meta_doc.exists else {}
    
    # Score all properties
    scored = []
    for i, prop in enumerate(index['props']):
        prop_vec = index['vectors'][i]
        sc = score_property_for_user(user_vec, prop_vec, prop, user_meta=user_meta)
        scored.append((prop['id'], sc['final_score'], sc))
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:count]
    
    # Prepare response
    results = []
    for pid, score, detail in top:
        prop_doc = db.collection('properties').document(pid).get()
        p = prop_doc.to_dict() if prop_doc.exists else {}
        p['id'] = pid
        p['recoScore'] = score
        p['recoDetails'] = detail
        results.append(p)
    
    return jsonify({"results": results})

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    print("Starting Flask recommendation server...")
    print(f"CORS enabled for all origins")
    print(f"Server running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)