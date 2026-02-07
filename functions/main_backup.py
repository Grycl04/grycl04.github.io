from firebase_admin import initialize_app, firestore, credentials
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy initialization
_firebase_app = None
_db_client = None

def get_firebase():
    """Lazy initialize Firebase"""
    global _firebase_app, _db_client
    if _firebase_app is None:
        logger.info("Initializing Firebase...")
        _firebase_app = initialize_app()
        _db_client = firestore.client()
    return _db_client

class FirestoreEncoder(json.JSONEncoder):
    """Custom JSON encoder for Firestore data"""
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)

# =========================== ENHANCED AI ALGORITHM ===========================

def get_user_profile(db, user_id: str):
    """Get comprehensive user profile including saved properties and search history"""
    profile = {
        "saved_properties": [],  # Property IDs the user has saved
        "search_history": [],    # Search filters used
        "search_categories": [], # Categories from searches
        "search_locations": [],  # Locations from searches
        "search_types": [],      # Property types from searches
        "transaction_prefs": [], # Transaction types (sale/rent/lease)
        "search_frequency": {},  # How often user searches for each category/location
        "saved_count": 0,
        "search_count": 0,
        "has_history": False,
        "saved_properties_details": []  # Detailed property data
    }
    
    try:
        # 1. GET SAVED PROPERTIES - SIMPLIFIED VERSION
        saved_ref = db.collection('savedProperties')
        saved_query = saved_ref.where('userId', '==', user_id)
        saved_snapshot = saved_query.get()
        
        valid_saved_ids = []
        profile["saved_properties_details"] = []
        
        for doc in saved_snapshot:
            data = doc.to_dict()
            property_id = data.get('propertyId')
            
            if property_id:
                valid_saved_ids.append(property_id)
                
                # Try to get property details
                try:
                    prop_doc = db.collection('properties').document(property_id).get()
                    if prop_doc.exists:
                        prop = prop_doc.to_dict()
                        prop['id'] = property_id
                        
                        # Add saved timestamp
                        if 'savedAt' in data:
                            prop['saved_timestamp'] = data['savedAt']
                        elif 'createdAt' in prop:
                            prop['saved_timestamp'] = prop['createdAt']
                        elif 'createdAt' in data:
                            prop['saved_timestamp'] = data['createdAt']
                        else:
                            prop['saved_timestamp'] = datetime.now()
                        
                        profile["saved_properties_details"].append(prop)
                    else:
                        # Property doesn't exist, but still count it
                        minimal_prop = {
                            'id': property_id,
                            'saved_timestamp': data.get('savedAt', data.get('createdAt', datetime.now())),
                            'is_active': False
                        }
                        profile["saved_properties_details"].append(minimal_prop)
                        
                except Exception as e:
                    logger.warning(f"Error getting property {property_id}: {str(e)}")
                    # Still count it as saved
                    minimal_prop = {
                        'id': property_id,
                        'saved_timestamp': data.get('savedAt', data.get('createdAt', datetime.now())),
                        'is_active': False
                    }
                    profile["saved_properties_details"].append(minimal_prop)
        
        # Update the saved properties list
        profile["saved_properties"] = valid_saved_ids
        
        # 2. GET SEARCH HISTORY
        events_ref = db.collection('events')
        
        # Get last 50 search events (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        try:
            search_query = (
                events_ref.where('userId', '==', user_id)
                .where('eventType', '==', 'search')
                .where('timestamp', '>=', thirty_days_ago)
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(50)
            )
            
            search_snapshot = search_query.get()
        except:
            # Fallback query without timestamp ordering
            search_query = (
                events_ref.where('userId', '==', user_id)
                .where('eventType', '==', 'search')
                .limit(50)
            )
            search_snapshot = search_query.get()
        
        search_patterns = []
        category_counter = defaultdict(int)
        location_counter = defaultdict(int)
        type_counter = defaultdict(int)
        transaction_counter = defaultdict(int)
        
        for doc in search_snapshot:
            data = doc.to_dict()
            filters = data.get('metadata', {}).get('filters', {})
            if filters:
                search_patterns.append(filters)
                
                # Extract and count search patterns
                if 'propertyCategory' in filters:
                    category = filters['propertyCategory']
                    category_counter[category] += 1
                    if category not in profile["search_categories"]:
                        profile["search_categories"].append(category)
                
                if 'location' in filters:
                    location = filters['location']
                    location_counter[location] += 1
                    if location not in profile["search_locations"]:
                        profile["search_locations"].append(location)
                
                if 'propertyType' in filters:
                    prop_type = filters['propertyType']
                    type_counter[prop_type] += 1
                    if prop_type not in profile["search_types"]:
                        profile["search_types"].append(prop_type)
                
                if 'transactionType' in filters:
                    trans_type = filters['transactionType']
                    transaction_counter[trans_type] += 1
                    if trans_type not in profile["transaction_prefs"]:
                        profile["transaction_prefs"].append(trans_type)
        
        profile["search_history"] = search_patterns
        profile["search_frequency"] = {
            "categories": dict(category_counter),
            "locations": dict(location_counter),
            "types": dict(type_counter),
            "transactions": dict(transaction_counter)
        }
        
        # 3. CALCULATE STATISTICS
        profile["saved_count"] = len(valid_saved_ids)
        profile["search_count"] = len(search_patterns)
        profile["has_history"] = profile["saved_count"] > 0 or profile["search_count"] > 0
        
        # 4. CALCULATE USER PREFERENCES
        preferences = {
            "primary_category": None,
            "primary_location": None,
            "primary_type": None,
            "primary_transaction": None,
            "confidence_score": 0
        }
        
        # Find most frequent searches
        if category_counter:
            preferences["primary_category"] = max(category_counter, key=category_counter.get)
        if location_counter:
            preferences["primary_location"] = max(location_counter, key=location_counter.get)
        if type_counter:
            preferences["primary_type"] = max(type_counter, key=type_counter.get)
        if transaction_counter:
            preferences["primary_transaction"] = max(transaction_counter, key=transaction_counter.get)
        
        # Calculate confidence based on search frequency and saved properties
        total_searches = sum(category_counter.values())
        if total_searches > 0:
            max_freq = max(category_counter.values()) if category_counter else 0
            preferences["confidence_score"] = min(100, int((max_freq / total_searches) * 100 * 0.7))
        
        if profile["saved_count"] > 0:
            preferences["confidence_score"] = min(100, preferences["confidence_score"] + (profile["saved_count"] * 10))
        
        profile["preferences"] = preferences
        
        logger.info(f"User {user_id[:8]}... profile: {profile['saved_count']} saves, {profile['search_count']} searches")
        
        return profile
        
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return {
            "saved_properties": [],
            "search_history": [],
            "search_categories": [],
            "search_locations": [],
            "search_types": [],
            "transaction_prefs": [],
            "search_frequency": {},
            "saved_count": 0,
            "search_count": 0,
            "has_history": False,
            "saved_properties_details": [],
            "preferences": {
                "primary_category": None,
                "primary_location": None,
                "primary_type": None,
                "primary_transaction": None,
                "confidence_score": 0
            }
        }

def calculate_property_score(property_data, user_profile, all_saved_ids):
    """Calculate comprehensive score for a property based on user profile"""
    
    # Initialize scores
    saved_similarity_score = 0
    search_match_score = 0
    preference_match_score = 0
    match_reasons = []
    
    # WEIGHTS
    WEIGHTS = {
        "saved_similarity": 0.5,
        "search_match": 0.3,
        "preference_match": 0.2
    }
    
    # 1. SIMILARITY TO SAVED PROPERTIES
    if user_profile["saved_count"] > 0 and "saved_properties_details" in user_profile:
        saved_details = user_profile["saved_properties_details"][:5]
        
        for saved_prop in saved_details:
            similarity = 0
            
            # Property type match
            if property_data.get('propertyType') == saved_prop.get('propertyType'):
                similarity += 30
                match_reasons.append(f"Same type as saved {saved_prop.get('propertyType', 'property')}")
            
            # Location match
            prop_location = property_data.get('city') or property_data.get('location') or ''
            saved_location = saved_prop.get('city') or saved_prop.get('location') or ''
            if prop_location and saved_location and prop_location.lower() == saved_location.lower():
                similarity += 25
                match_reasons.append(f"Same location as saved property in {prop_location}")
            
            # Price range match
            prop_price = property_data.get('monthlyRent') or property_data.get('pricing') or property_data.get('salePrice') or 0
            saved_price = saved_prop.get('monthlyRent') or saved_prop.get('pricing') or saved_prop.get('salePrice') or 0
            
            if prop_price > 0 and saved_price > 0:
                price_ratio = min(prop_price, saved_price) / max(prop_price, saved_price)
                if price_ratio >= 0.6:
                    similarity += 20
                    match_reasons.append(f"Similar price to saved properties")
                elif price_ratio >= 0.3:
                    similarity += 10
            
            # Bedrooms match
            if property_data.get('bedrooms') == saved_prop.get('bedrooms'):
                similarity += 15
                match_reasons.append(f"Same number of bedrooms as saved property")
            
            # Property category match
            if property_data.get('propertyCategory') == saved_prop.get('propertyCategory'):
                similarity += 10
            
            saved_similarity_score += similarity / len(saved_details)
    
    # 2. MATCH WITH SEARCH HISTORY
    if user_profile["search_count"] > 0:
        search_bonus = 0
        
        # Category match
        prop_category = property_data.get('propertyCategory')
        if prop_category and prop_category in user_profile["search_categories"]:
            frequency = user_profile["search_frequency"]["categories"].get(prop_category, 0)
            search_bonus += 30 + (frequency * 5)
            match_reasons.append(f"Matches your frequently searched category: {prop_category}")
        
        # Location match
        prop_location = property_data.get('city') or property_data.get('location') or ''
        for loc in user_profile["search_locations"]:
            if loc and prop_location and loc.lower() in prop_location.lower():
                frequency = user_profile["search_frequency"]["locations"].get(loc, 0)
                search_bonus += 25 + (frequency * 5)
                match_reasons.append(f"Located in {loc} which you've searched for")
                break
        
        # Property type match
        prop_type = property_data.get('propertyType')
        if prop_type and prop_type in user_profile["search_types"]:
            frequency = user_profile["search_frequency"]["types"].get(prop_type, 0)
            search_bonus += 20 + (frequency * 5)
            match_reasons.append(f"Type matches your searches: {prop_type}")
        
        # Transaction type match
        prop_transaction = 'sale'
        if property_data.get('monthlyRent'):
            prop_transaction = 'rent'
        elif property_data.get('annualRent'):
            prop_transaction = 'lease'
        
        if prop_transaction in user_profile["transaction_prefs"]:
            frequency = user_profile["search_frequency"]["transactions"].get(prop_transaction, 0)
            search_bonus += 15 + (frequency * 5)
            match_reasons.append(f"Matches your preferred transaction type: {prop_transaction}")
        
        search_match_score = min(100, search_bonus)
    
    # 3. MATCH WITH IDENTIFIED PREFERENCES
    preferences = user_profile["preferences"]
    if preferences["confidence_score"] > 50:
        preference_bonus = 0
        
        if preferences["primary_category"] and property_data.get('propertyCategory') == preferences["primary_category"]:
            preference_bonus += 40
            match_reasons.append(f"Matches your primary interest: {preferences['primary_category']}")
        
        prop_location = property_data.get('city') or property_data.get('location') or ''
        if preferences["primary_location"] and preferences["primary_location"].lower() in prop_location.lower():
            preference_bonus += 35
            match_reasons.append(f"Located in your preferred area: {preferences['primary_location']}")
        
        if preferences["primary_type"] and property_data.get('propertyType') == preferences["primary_type"]:
            preference_bonus += 25
            match_reasons.append(f"Type matches your preference: {preferences['primary_type']}")
        
        preference_match_score = min(100, preference_bonus)
    
    # 4. FINAL SCORE CALCULATION
    final_score = (
        saved_similarity_score * WEIGHTS["saved_similarity"] +
        search_match_score * WEIGHTS["search_match"] +
        preference_match_score * WEIGHTS["preference_match"]
    )
    
    match_count = len(set(match_reasons))
    if match_count > 1:
        final_score *= (1 + (match_count * 0.1))
    
    final_score = min(100, final_score)
    
    unique_reasons = list(dict.fromkeys(match_reasons))
    if not unique_reasons:
        if user_profile["saved_count"] > 0:
            unique_reasons = ["Similar to properties you've saved"]
        else:
            unique_reasons = ["Based on your search patterns"]
    
    return {
        "score": final_score,
        "match_reasons": unique_reasons[:3],
        "component_scores": {
            "saved_similarity": saved_similarity_score,
            "search_match": search_match_score,
            "preference_match": preference_match_score
        }
    }

def find_ai_recommendations(db, user_profile, all_properties, count: int = 5):
    """Find AI-powered recommendations based on comprehensive user profile"""
    try:
        scored_properties = []
        saved_ids = set(user_profile["saved_properties"])
        
        for prop in all_properties:
            if prop['id'] in saved_ids:
                continue
            
            score_result = calculate_property_score(prop, user_profile, saved_ids)
            
            if score_result["score"] > 20:
                prop['similarity_score'] = score_result["score"]
                prop['match_score'] = int(score_result["score"])
                prop['match_reason'] = score_result["match_reasons"][0] if score_result["match_reasons"] else "Recommended for you"
                prop['match_details'] = score_result["match_reasons"]
                prop['is_ai_recommended'] = True
                scored_properties.append(prop)
        
        scored_properties.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        final_recommendations = []
        seen_categories = defaultdict(int)
        seen_locations = defaultdict(int)
        
        for prop in scored_properties:
            prop_category = prop.get('propertyCategory', 'unknown')
            prop_location = prop.get('city') or prop.get('location') or 'unknown'
            
            if seen_categories[prop_category] < 2 and seen_locations[prop_location] < 2:
                final_recommendations.append(prop)
                seen_categories[prop_category] += 1
                seen_locations[prop_location] += 1
            
            if len(final_recommendations) >= count:
                break
        
        logger.info(f"Found {len(final_recommendations)} AI recommendations")
        return final_recommendations
        
    except Exception as e:
        logger.error(f"Error finding AI recommendations: {str(e)}")
        return []

def find_search_based_recommendations(db, user_profile, all_properties, count: int = 5):
    """Find recommendations based primarily on search history"""
    try:
        scored_properties = []
        
        for prop in all_properties:
            search_bonus = 0
            match_reasons = []
            
            prop_category = prop.get('propertyCategory')
            if prop_category and prop_category in user_profile["search_categories"]:
                frequency = user_profile["search_frequency"]["categories"].get(prop_category, 0)
                search_bonus += 40 + (frequency * 10)
                match_reasons.append(f"Category you search often: {prop_category}")
            
            prop_location = prop.get('city') or prop.get('location') or ''
            for loc in user_profile["search_locations"]:
                if loc and prop_location and loc.lower() in prop_location.lower():
                    frequency = user_profile["search_frequency"]["locations"].get(loc, 0)
                    search_bonus += 35 + (frequency * 10)
                    match_reasons.append(f"Area you've searched: {loc}")
                    break
            
            prop_type = prop.get('propertyType')
            if prop_type and prop_type in user_profile["search_types"]:
                frequency = user_profile["search_frequency"]["types"].get(prop_type, 0)
                search_bonus += 25 + (frequency * 8)
                match_reasons.append(f"Type you look for: {prop_type}")
            
            if search_bonus > 30:
                prop['similarity_score'] = min(100, search_bonus)
                prop['match_score'] = int(min(100, search_bonus))
                prop['match_reason'] = match_reasons[0] if match_reasons else "Based on your searches"
                prop['match_details'] = match_reasons
                prop['is_search_based'] = True
                scored_properties.append(prop)
        
        scored_properties.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return scored_properties[:count]
        
    except Exception as e:
        logger.error(f"Error finding search-based recommendations: {str(e)}")
        return []

def get_recent_properties(db, count: int = 5):
    """Get recent properties as fallback"""
    try:
        properties_ref = db.collection('properties')
        
        try:
            query = properties_ref.where('status', 'in', ['active', 'available', 'Active']) \
                                 .order_by('createdAt', direction=firestore.Query.DESCENDING) \
                                 .limit(count * 2)
        except:
            query = properties_ref.where('status', 'in', ['active', 'available', 'Active']) \
                                 .limit(count * 2)
        
        snapshot = query.get()
        
        properties = []
        for doc in snapshot:
            prop = doc.to_dict()
            prop['id'] = doc.id
            properties.append(prop)
        
        return properties[:count]
        
    except Exception as e:
        logger.error(f"Error getting recent properties: {str(e)}")
        return []

# =========================== CLOUD FUNCTION ===========================

@functions_framework.http
def personalized_recommendations(request):
    """Enhanced AI recommendations considering saved properties AND search history"""
    
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600"
        }
        return ("", 204, headers)
    
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return (json.dumps({
                "success": False, 
                "error": "user_id is required",
                "message": "User ID is required"
            }), 400, {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})
        
        count = int(request.args.get('count', 5))
        
        logger.info(f"Getting ENHANCED recommendations for user: {user_id[:8]}...")
        
        db = get_firebase()
        
        user_profile = get_user_profile(db, user_id)
        
        all_active_props = []
        try:
            props_ref = db.collection('properties')
            props_query = props_ref.where('status', 'in', ['active', 'available', 'Active'])
            props_snapshot = props_query.get()
            
            for doc in props_snapshot:
                prop = doc.to_dict()
                prop['id'] = doc.id
                all_active_props.append(prop)
            
            logger.info(f"Found {len(all_active_props)} active properties")
        except Exception as e:
            logger.error(f"Error getting active properties: {str(e)}")
            all_active_props = []
        
        if not user_profile["has_history"]:
            recommendations = get_recent_properties(db, count)
            
            response = {
                "success": True,
                "user_id": user_id,
                "has_history": False,
                "recommendation_type": "recent",
                "message": "Save properties or search for properties to get personalized AI recommendations!",
                "user_message": "Start by browsing properties and saving ones you're interested in. Save 3+ properties to unlock AI recommendations.",
                "saved_count": 0,
                "search_count": 0,
                "required_for_ai": 3,
                "recommendations": recommendations,
                "count": len(recommendations),
                "profile_summary": {
                    "confidence": 0,
                    "primary_interests": "No data yet"
                },
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"User {user_id[:8]}... has no history - returning {len(recommendations)} recent properties")
        
        elif user_profile["saved_count"] < 3:
            if user_profile["search_count"] >= 3:
                recommendations = find_search_based_recommendations(db, user_profile, all_active_props, count)
                rec_type = "search_based"
                message = f"Based on your {user_profile['search_count']} recent searches"
                user_message = f"Based on your recent searches, here are properties that match what you're looking for. Save ones you like to improve recommendations!"
            else:
                recommendations = get_recent_properties(db, count)
                rec_type = "recent"
                message = f"You've saved {user_profile['saved_count']} properties and made {user_profile['search_count']} searches"
                user_message = f"Save more properties or search for specific types to get better recommendations!"
            
            profile_summary = {
                "confidence": user_profile["preferences"]["confidence_score"],
                "primary_interests": []
            }
            
            prefs = user_profile["preferences"]
            if prefs["primary_category"]:
                profile_summary["primary_interests"].append(prefs["primary_category"])
            if prefs["primary_location"]:
                profile_summary["primary_interests"].append(f"Location: {prefs['primary_location']}")
            
            if not profile_summary["primary_interests"]:
                profile_summary["primary_interests"] = ["Exploring options"]
            
            response = {
                "success": True,
                "user_id": user_id,
                "has_history": True,
                "recommendation_type": rec_type,
                "message": message,
                "user_message": user_message,
                "saved_count": user_profile["saved_count"],
                "search_count": user_profile["search_count"],
                "required_for_ai": 3,
                "progress_percentage": int((user_profile["saved_count"] / 3) * 100),
                "recommendations": recommendations,
                "count": len(recommendations),
                "profile_summary": profile_summary,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"User {user_id[:8]}... has {user_profile['saved_count']} saves, {user_profile['search_count']} searches - returning {len(recommendations)} {rec_type} recommendations")
        
        else:
            recommendations = find_ai_recommendations(db, user_profile, all_active_props, count)
            
            if len(recommendations) < 3:
                logger.info(f"AI found only {len(recommendations)} recommendations, supplementing with search-based")
                search_based = find_search_based_recommendations(db, user_profile, all_active_props, count - len(recommendations))
                
                existing_ids = {r['id'] for r in recommendations}
                for rec in search_based:
                    if rec['id'] not in existing_ids and len(recommendations) < count:
                        rec['is_supplemental'] = True
                        recommendations.append(rec)
            
            if len(recommendations) < 3:
                recent = get_recent_properties(db, count - len(recommendations))
                existing_ids = {r['id'] for r in recommendations}
                for rec in recent:
                    if rec['id'] not in existing_ids and len(recommendations) < count:
                        rec['is_fallback'] = True
                        recommendations.append(rec)
            
            prefs = user_profile["preferences"]
            profile_summary = {
                "confidence": prefs["confidence_score"],
                "primary_category": prefs["primary_category"],
                "primary_location": prefs["primary_location"],
                "primary_type": prefs["primary_type"],
                "primary_transaction": prefs["primary_transaction"],
                "search_habits": {
                    "total_searches": user_profile["search_count"],
                    "frequent_categories": list(user_profile["search_frequency"]["categories"].keys())[:3],
                    "frequent_locations": list(user_profile["search_frequency"]["locations"].keys())[:3]
                }
            }
            
            if user_profile["saved_count"] >= 5:
                match_strategy = "primarily based on your saved properties"
            elif user_profile["search_count"] >= 10:
                match_strategy = "based on your search patterns and saved properties"
            else:
                match_strategy = "based on your preferences"
            
            response = {
                "success": True,
                "user_id": user_id,
                "has_history": True,
                "recommendation_type": "ai",
                "message": f"AI found {len(recommendations)} properties {match_strategy}",
                "user_message": f"Based on your {user_profile['saved_count']} saved properties and {user_profile['search_count']} searches, here are personalized recommendations",
                "saved_count": user_profile["saved_count"],
                "search_count": user_profile["search_count"],
                "recommendations": recommendations,
                "count": len(recommendations),
                "profile_summary": profile_summary,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"User {user_id[:8]}... - returning {len(recommendations)} AI recommendations (confidence: {prefs['confidence_score']}%)")
        
        return (json.dumps(response, cls=FirestoreEncoder, indent=2),
                200,
                {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})
        
    except Exception as e:
        logger.error(f"Error in personalized recommendations: {str(e)}", exc_info=True)
        return (json.dumps({
            "success": False, 
            "error": str(e),
            "recommendation_type": "error",
            "message": "Unable to generate recommendations. Please try again."
        }), 500, {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})

@functions_framework.http
def user_profile_insights(request):
    """Get insights about user's preferences based on saved properties and searches"""
    
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600"
        }
        return ("", 204, headers)
    
    user_id = request.args.get('user_id')
    if not user_id:
        return (json.dumps({"error": "user_id required"}),
                400,
                {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})
    
    try:
        db = get_firebase()
        profile = get_user_profile(db, user_id)
        
        insights = []
        
        if profile["saved_count"] > 0:
            insights.append(f"You've saved {profile['saved_count']} properties")
        
        if profile["search_count"] > 0:
            insights.append(f"You've made {profile['search_count']} searches in the last 30 days")
        
        prefs = profile["preferences"]
        if prefs["primary_category"]:
            insights.append(f"You're most interested in: {prefs['primary_category']}")
        
        if prefs["primary_location"]:
            insights.append(f"You frequently search in: {prefs['primary_location']}")
        
        if prefs["primary_type"]:
            insights.append(f"You prefer: {prefs['primary_type']}")
        
        if not insights:
            insights = ["Start saving properties or searching to build your profile"]
        
        response = {
            "user_id": user_id,
            "profile": {
                "saved_count": profile["saved_count"],
                "search_count": profile["search_count"],
                "confidence_score": prefs["confidence_score"],
                "preferences": prefs,
                "frequent_categories": dict(sorted(
                    profile["search_frequency"]["categories"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]),
                "frequent_locations": dict(sorted(
                    profile["search_frequency"]["locations"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            },
            "insights": insights,
            "recommendations": {
                "save_more": max(0, 3 - profile["saved_count"]),
                "search_more": "Try specific searches to improve recommendations",
                "ai_ready": profile["saved_count"] >= 3 or profile["search_count"] >= 10
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return (json.dumps(response, cls=FirestoreEncoder, indent=2),
                200,
                {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})
    except Exception as e:
        logger.error(f"Error getting user insights: {str(e)}")
        return (json.dumps({"error": str(e)}),
                500,
                {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})

@functions_framework.http
def health(request):
    """Health check"""
    return (json.dumps({
        "status": "healthy",
        "service": "bahai-enhanced-recommender",
        "version": "3.1",  # Updated version
        "timestamp": datetime.now().isoformat(),
        "algorithm_features": {
            "saved_property_analysis": True,
            "search_history_analysis": True,
            "preference_extraction": True,
            "weighted_scoring": True,
            "diversity_filtering": True,
            "confidence_scoring": True
        },
        "data_sources": {
            "saved_properties": "Primary signal",
            "search_filters": "Secondary signal"
        }
    }, indent=2),
    200,
    {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"})