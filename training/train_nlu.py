import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import glob
import pandas as pd
import numpy as np
import re
from collections import Counter
import logging
import random
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamNLUTrainer:
    def __init__(self):
        # Try to load spaCy, fallback if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except:
            logger.warning("⚠️ spaCy model not found. Using basic preprocessing.")
            self.nlp = None
        
        # Create pipeline with improved parameters
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=2500,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )),
            ('classifier', SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                C=1.0,
                class_weight='balanced'
            ))
        ])
        
        # Team member assignments
        self.team_assignments = {
            'member1': ['find_property', 'financing', 'location_info'],
            'member2': ['find_property_with_criteria', 'find_near_landmark', 'find_ready_property'],
            'member3': ['find_property_for_need', 'find_with_feature', 'process_info', 'match_needs']
        }
        
        # Template to intent mapping
        self.template_intent_map = {
            'question_1': 'find_property',
            'question_2': 'find_property_with_criteria',
            'question_3': 'find_property_for_need',
            'question_4': 'find_near_landmark',
            'question_5': 'find_with_feature',
            'question_6': 'find_ready_property',
            'question_7': 'financing',
            'question_8': 'process_info',
            'question_9': 'location_info',
            'question_10': 'match_needs'
        }
        
        # Intent mapping from old names to standard names
        self.intent_mapping = {
            'type_price_features': 'find_property_with_criteria',
            'near_landmark': 'find_near_landmark',
            'ready_to_move': 'find_ready_property',
            'family_needs': 'find_property_for_need',
            'feature_price': 'find_with_feature',
            'process_info': 'process_info',
            'personalized_match': 'match_needs',
            'location_info': 'location_info',
            'financing_info': 'financing',
            'financing': 'financing',
            'find_property': 'find_property',
            'greeting': 'greeting',  
            'help': 'help',          
            'thanks': 'thanks'       
        }
        
        # Intent keywords for better classification
        self.intent_keywords = {
            'greeting': ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon'],
            'help': ['help', 'what can you do', 'how can you help', 'assist', 'support'],
            'thanks': ['thank you', 'thanks', 'appreciate', 'grateful'],
            'financing': ['accept bank financing', 'accept financing', 'bank loan', 
                         'mortgage', 'pag-ibig', 'payment method', 'financing type',
                         'documents needed', 'requirements for', 'how to get',
                         'what documents', 'loan requirements', 'bank financing'],
            'find_ready_property': ['ready to move in', 'ready for occupancy', 
                                   'available now', 'immediate occupancy', 
                                   'move in ready', 'ready now', 'ready to occupy',
                                   'immediate move in', 'available immediately'],
            'process_info': ['steps for', 'how to', 'process of', 'procedure', 
                            'timeline', 'requirements', 'documents', 'steps to',
                            'how do i', 'what are the steps', 'costs for',
                            'timeline for', 'process for'],
            'find_with_feature': ['with swimming pool', 'with pool', 'with garden', 
                                 'with parking', 'with elevator', 'with security',
                                 'with wifi', 'with furniture', 'with aircon',
                                 'with feature', 'featuring', 'having'],
            'find_near_landmark': ['near schools', 'near mall', 'near hospital', 
                                  'near port', 'near beach', 'near church',
                                  'near landmark', 'close to', 'around',
                                  'beside', 'next to', 'adjacent to'],
            'location_info': ['tell me about', 'what is', 'describe', 'about the',
                             'information about', 'living in', 'like to live',
                             'what\'s it like', 'is it good', 'lifestyle'],
            'find_property': ['find', 'search for', 'show me', 'looking for',
                             'need', 'want', 'locate', 'discover', 
                              'what apartments', 'what houses', 'what condos',  
                     'do you have', 'any properties', 'available properties'], 
            'find_property_for_need': ['for family', 'for students', 'for professionals',
                                      'for couple', 'for retirees', 'for business',
                                      'for investors', 'for single', 'for workers'],
             'find_property_with_criteria': [
        'under', 'below', 'less than', 'maximum', 'up to',
        'with bedroom', 'with bath', 'with bathrooms',
        'with bedrooms', 'bedroom', 'bathroom', 'rooms',
        'price range', 'budget', 'affordable', 'cheap'
    ],
            'match_needs': ['match my', 'suitable for', 'fitting my', 'appropriate for',
                           'compatible with', 'what matches', 'recommendations for']
        }
        
        # Load Batangas data for location training
        self.batangas_data = self.load_batangas_data()

    def load_batangas_data(self):
        """Load Batangas complete data for location-based training"""
        batangas_file = 'data/shared/batangas_complete.json'
        if not os.path.exists(batangas_file):
            logger.warning(f"⚠️ Batangas data file not found: {batangas_file}")
            return {}
        
        try:
            with open(batangas_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("✅ Batangas data loaded successfully")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading Batangas data: {e}")
            return {}

    def clean_json_file(self, filepath):
        """Fix JSON file by properly loading and saving it"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove any trailing commas before closing braces/brackets
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            
            # Parse the JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to fix by finding the problematic section
                lines = content.split('\n')
                cleaned_lines = []
                for line in lines:
                    if '//' in line:
                        line = line.split('//')[0]
                    cleaned_lines.append(line.strip())
                content = '\n'.join(cleaned_lines)
                data = json.loads(content)
            
            # Write it back with proper formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Cleaned {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cleaning {filepath}: {e}")
            return False

    def preprocess_text(self, text):
        """Preprocess text for training with keyword preservation"""
        if not text:
            return ""
        
        original_text = text.lower()
        text = str(text).lower()
        
        # First, preserve important intent patterns by marking them
        text = self.mark_intent_keywords(text, original_text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\?\.\-\:]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If spaCy is loaded, do lemmatization
        if self.nlp:
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    tokens.append(token.lemma_)
            return ' '.join(tokens)
        
        return text
    
    def mark_intent_keywords(self, text, original_text):
        """Mark intent keywords in the text"""
        marked_text = text
        
        # Check each intent for keywords
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in original_text:
                    # Replace with marked version
                    marked_text = marked_text.replace(keyword, f"{keyword}_INTENT_{intent}")
        
        return marked_text

    def load_member_data(self, base_path='data'):
        """Load training data from all team members"""
        texts = []
        intents = []
        
        member_files = glob.glob(os.path.join(base_path, 'member*', 'training_data.json'))
        
        if not member_files:
            logger.warning("❌ No member training files found!")
            return texts, intents
        
        for member_file in member_files:
            member_name = os.path.basename(os.path.dirname(member_file))
            print(f"📂 Loading {member_name} data...")
            
            # Clean the JSON file first
            self.clean_json_file(member_file)
            
            try:
                with open(member_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                samples = data.get('training_samples', [])
                
                for sample in samples:
                    # Get intent and map to standard name
                    original_intent = sample.get('intent', '')
                    mapped_intent = self.intent_mapping.get(original_intent, original_intent)
                    
                    # Main query
                    query = sample.get('query', '').strip()
                    if query:
                        texts.append(self.preprocess_text(query))
                        intents.append(mapped_intent)
                    
                    # Variations
                    variations = sample.get('variations', [])
                    for variation in variations:
                        if isinstance(variation, str) and variation.strip():
                            texts.append(self.preprocess_text(variation))
                            intents.append(mapped_intent)
                
                print(f"   ✅ Loaded {len(samples)} samples from {member_name}")
                
            except Exception as e:
                print(f"   ❌ Error loading {member_file}: {e}")
        
        return texts, intents

    def add_corrective_training_samples(self):
        """Add specific training samples to fix common misclassifications"""
        print("\n🔧 Adding corrective training samples...")
        
        corrective_samples = [
            # Clear examples of find_property_with_criteria
            ("houses under 30 million with 4 bedrooms", "find_property_with_criteria"),
            ("show me properties below 20M with 3 bedrooms", "find_property_with_criteria"),
            ("find condos under 10M with 2 baths", "find_property_with_criteria"),
            
            # Clear examples of find_property (no price/bedroom criteria)
            ("find houses in nasugbu", "find_property"),
            ("show me apartments in batangas city", "find_property"),
            ("look for condos in lipa", "find_property"),
            
            # Boundary cases
            ("houses with 4 bedrooms", "find_property_with_criteria"),  # has bedroom count
            ("houses under 30M", "find_property_with_criteria"),  # has price
            ("houses in batangas", "find_property"),  # only location
            
            # General property searches WITHOUT location
            ("find apartments", "find_property"),
            ("show me houses", "find_property"),
            ("look for condos", "find_property"),
            ("search for properties", "find_property"),
            ("i need a house", "find_property"),
            ("show me available properties", "find_property"),
            ("find beachfront properties", "find_property"),
            ("show me commercial spaces", "find_property"),
            ("looking for townhouses", "find_property"),
            ("need a studio", "find_property"),
            
            # General property type questions
            ("what apartments do you have", "find_property"),
            ("what houses are available", "find_property"),
            ("show me all condos", "find_property"),
            ("what properties do you offer", "find_property"),
            ("do you have any apartments", "find_property"),
            ("any houses for sale", "find_property"),
            ("any condos for rent", "find_property"),
            
            # Financing intent fixes
            ("properties that accept bank financing", "financing"),
            ("show me properties that accept bank financing", "financing"),
            ("houses that accept bank loans", "financing"),
            ("properties with bank financing options", "financing"),
            ("real estate that accepts pag-ibig", "financing"),
            ("condos with in-house financing", "financing"),
            ("how to get bank financing for a house", "financing"),
            ("what documents for pag-ibig loan", "financing"),
            ("bank financing requirements", "financing"),
            ("properties accepting cash payment", "financing"),
            
            # Ready to move property fixes
            ("find ready to move in properties for students in batangas city", "find_ready_property"),
            ("ready to occupy apartments for students", "find_ready_property"),
            ("available now properties for family", "find_ready_property"),
            ("immediate occupancy houses", "find_ready_property"),
            ("move in ready condos for professionals", "find_ready_property"),
            ("properties ready now for couples", "find_ready_property"),
            ("ready for occupancy commercial spaces", "find_ready_property"),
            ("available immediately near schools", "find_ready_property"),
            ("ready to move in with furniture", "find_ready_property"),
            ("properties ready for move in", "find_ready_property"),
            
            # Process info fixes
            ("steps for buying a condo", "process_info"),
            ("how to buy a house step by step", "process_info"),
            ("process of purchasing property", "process_info"),
            ("timeline for renting an apartment", "process_info"),
            ("requirements for commercial space lease", "process_info"),
            ("documents needed for house purchase", "process_info"),
            ("procedure for getting a mortgage", "process_info"),
            ("what are the steps to invest in real estate", "process_info"),
            ("how does the property buying process work", "process_info"),
            ("steps costs timeline for townhouse", "process_info"),
            
            # With feature fixes
            ("properties with swimming pool", "find_with_feature"),
            ("houses with garden", "find_with_feature"),
            ("apartments with parking space", "find_with_feature"),
            ("condos with security", "find_with_feature"),
            ("properties featuring pool", "find_with_feature"),
            ("homes with private pool", "find_with_feature"),
            ("units with elevator", "find_with_feature"),
            ("properties with wifi included", "find_with_feature"),
            ("houses with home office", "find_with_feature"),
            ("apartments with furniture", "find_with_feature"),
            
            # Near landmark fixes
            ("properties near schools", "find_near_landmark"),
            ("houses close to malls", "find_near_landmark"),
            ("apartments near hospitals", "find_near_landmark"),
            ("condos near batangas port", "find_near_landmark"),
            ("properties around universities", "find_near_landmark"),
            ("real estate near beaches", "find_near_landmark"),
            ("housing near industrial parks", "find_near_landmark"),
            ("properties adjacent to churches", "find_near_landmark"),
            ("homes near business districts", "find_near_landmark"),
            ("apartments near transport hubs", "find_near_landmark"),
            
            # Location info fixes
            ("tell me about batangas city", "location_info"),
            ("what is lipa city like", "location_info"),
            ("describe tanauan city", "location_info"),
            ("information about nasugbu", "location_info"),
            ("living in san juan batangas", "location_info"),
            ("about calatagan", "location_info"),
            ("what's it like to live in taal", "location_info"),
            ("describe mabini batangas", "location_info"),
            ("is sto tomas a good place to live", "location_info"),
            ("tell me about the lifestyle in malvar", "location_info"),
        ]
        
        texts = []
        intents = []
        
        for text, intent in corrective_samples:
            texts.append(self.preprocess_text(text))
            intents.append(intent)
        
        print(f"   ✅ Added {len(texts)} corrective samples")
        return texts, intents
    
    def load_shared_questions(self, shared_path='data/shared'):
        """Load question templates from all_questions.json"""
        texts = []
        intents = []
        
        questions_file = os.path.join(shared_path, 'all_questions.json')
        
        if not os.path.exists(questions_file):
            print(f"❌ Shared questions file not found: {questions_file}")
            return texts, intents
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            question_templates = data.get('question_templates', {})
            print(f"📂 Loading {len(question_templates)} question templates...")
            
            templates_loaded = 0
            
            for q_id, q_data in question_templates.items():
                # Get intent from mapping
                intent = self.template_intent_map.get(q_id, 'unknown')
                
                # Add the example query
                example = q_data.get('example', '')
                if example:
                    texts.append(self.preprocess_text(example))
                    intents.append(intent)
                    templates_loaded += 1
                
                # Add templates
                template = q_data.get('template', '')
                if template:
                    texts.append(self.preprocess_text(template))
                    intents.append(intent)
                    templates_loaded += 1
            
            print(f"   ✅ Generated {templates_loaded} samples from templates")
            
        except Exception as e:
            print(f"   ❌ Error loading questions file: {e}")
        
        return texts, intents

    def load_synonyms_as_training(self, shared_path='data/shared'):
        """Load synonyms and generate training samples"""
        texts = []
        intents = []
        
        synonyms_file = os.path.join(shared_path, 'synonyms.json')
        
        if not os.path.exists(synonyms_file):
            print(f"❌ Synonyms file not found: {synonyms_file}")
            return texts, intents
        
        try:
            with open(synonyms_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("📂 Loading synonyms data...")
            
            # Map phrase categories to intents
            phrase_intent_map = {
                'property_search': 'find_property',
                'price_inquiry': 'financing',
                'location_specific': 'location_info',
                'feature_requests': 'find_with_feature',
                'process_questions': 'process_info',
            }
            
            # Use phrases section
            phrases = data.get('phrases', {})
            for category, phrase_list in phrases.items():
                if isinstance(phrase_list, list):
                    intent = phrase_intent_map.get(category, 'find_property')
                    for phrase in phrase_list[:5]:
                        if isinstance(phrase, str) and phrase.strip():
                            texts.append(self.preprocess_text(phrase))
                            intents.append(intent)
            
            print(f"   ✅ Generated {len(texts)} samples from synonyms")
            
        except Exception as e:
            print(f"   ❌ Error loading synonyms: {e}")
        
        return texts, intents

    def load_batangas_training(self):
        """Generate training data from Batangas complete data"""
        texts = []
        intents = []
        
        if not self.batangas_data:
            return texts, intents
        
        print("📂 Loading Batangas location data for training...")
        
        # Get locations from batangas data
        locations = self.batangas_data.get('batangas_locations', {})
        
        # Generate location-specific queries
        for location_name, location_data in locations.items():
            if isinstance(location_name, str):
                loc_name = location_name.lower()
                
                # Location info queries
                texts.append(f"tell me about {loc_name}")
                intents.append('location_info')
                
                texts.append(f"what is {loc_name} like")
                intents.append('location_info')
                
                # Find property queries
                texts.append(f"find properties in {loc_name}")
                intents.append('find_property')
                
                texts.append(f"show me houses in {loc_name}")
                intents.append('find_property')
        
        print(f"   ✅ Generated {len(texts)} samples from Batangas data")
        return texts, intents

    def load_additional_training(self, filepath='data/additional_training.json'):
        """Load additional training data"""
        texts = []
        intents = []
        
        if not os.path.exists(filepath):
            return texts, intents
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            additional_samples = data.get('additional_samples', [])
            for sample in additional_samples:
                text = sample.get('text', '').strip()
                intent = sample.get('intent', '').strip()
                if text and intent:
                    texts.append(self.preprocess_text(text))
                    intents.append(intent)
            
            logger.info(f"✅ Loaded {len(additional_samples)} additional samples")
        except Exception as e:
            logger.error(f"❌ Error loading additional training: {e}")
        
        return texts, intents

    def generate_additional_variations(self, texts, intents):
        """Generate additional variations for training"""
        new_texts = []
        new_intents = []
        
        # Limit to avoid too many samples
        limit = min(50, len(texts))
        
        for i in range(limit):
            text = texts[i]
            intent = intents[i]
            
            # Add question variation
            if not text.endswith('?'):
                new_texts.append(text + '?')
                new_intents.append(intent)
            
            # Add "please" variation
            new_texts.append('please ' + text)
            new_intents.append(intent)
            
            # Add "can you" variation
            new_texts.append('can you ' + text)
            new_intents.append(intent)
            
            # Add "i need" variation
            new_texts.append('i need ' + text)
            new_intents.append(intent)
        
        return new_texts, new_intents

    def load_all_training_data(self, base_path='data'):
        """Load ALL training data from all sources"""
        all_texts = []
        all_intents = []
        
        print("="*60)
        print("🚀 LOADING ALL TRAINING DATA SOURCES")
        print("="*60)
        
        # 1. Load member data
        print("\n📁 Source 1: Member Training Data")
        member_texts, member_intents = self.load_member_data(base_path)
        all_texts.extend(member_texts)
        all_intents.extend(member_intents)
        print(f"   ✅ Loaded {len(member_texts)} samples")
        print(f"   Total so far: {len(all_texts)} samples")
        
        # 2. Add corrective training samples (FIX for your issues)
        print("\n📁 Source 2: Corrective Training Samples")
        corrective_texts, corrective_intents = self.add_corrective_training_samples()
        all_texts.extend(corrective_texts)
        all_intents.extend(corrective_intents)
        print(f"   ✅ Added {len(corrective_texts)} corrective samples")
        print(f"   Total so far: {len(all_texts)} samples")
        
        # 3. Load shared questions
        print("\n📁 Source 3: Shared Question Templates")
        shared_path = os.path.join(base_path, 'shared')
        question_texts, question_intents = self.load_shared_questions(shared_path)
        all_texts.extend(question_texts)
        all_intents.extend(question_intents)
        print(f"   ✅ Generated {len(question_texts)} samples")
        print(f"   Total so far: {len(all_texts)} samples")
        
        # 4. Load synonyms as training data
        print("\n📁 Source 4: Synonyms and Phrases")
        synonym_texts, synonym_intents = self.load_synonyms_as_training(shared_path)
        all_texts.extend(synonym_texts)
        all_intents.extend(synonym_intents)
        print(f"   ✅ Generated {len(synonym_texts)} samples")
        print(f"   Total so far: {len(all_texts)} samples")
        
        # 5. Load Batangas data for training
        print("\n📁 Source 5: Batangas Location Data")
        batangas_texts, batangas_intents = self.load_batangas_training()
        all_texts.extend(batangas_texts)
        all_intents.extend(batangas_intents)
        print(f"   ✅ Generated {len(batangas_texts)} samples")
        print(f"   Total so far: {len(all_texts)} samples")
        
        # 6. Load additional training data
        print("\n📁 Source 6: Additional Training Data")
        additional_texts, additional_intents = self.load_additional_training()
        all_texts.extend(additional_texts)
        all_intents.extend(additional_intents)
        if additional_texts:
            print(f"   ✅ Loaded {len(additional_texts)} samples")
        print(f"   Total so far: {len(all_texts)} samples")
        
        # 7. Generate additional variations
        print("\n📁 Source 7: Generated Variations")
        generated_texts, generated_intents = self.generate_additional_variations(all_texts, all_intents)
        all_texts.extend(generated_texts)
        all_intents.extend(generated_intents)
        print(f"   ✅ Generated {len(generated_texts)} variations")
        
        print("="*60)
        print(f"📊 FINAL TRAINING DATA STATISTICS")
        print("="*60)
        print(f"✅ Total samples: {len(all_texts)}")
        
        # Count unique intents
        unique_intents = set(all_intents)
        print(f"✅ Unique intents: {len(unique_intents)}")
        
        # Count intent distribution
        intent_counts = Counter(all_intents)
        print(f"✅ Intent distribution:")
        for intent, count in intent_counts.most_common():
            print(f"   • {intent}: {count} samples")
        
        return all_texts, all_intents

    def train(self, training_texts, training_intents):
        """Train the NLU model with class balancing"""
        if not training_texts:
            logger.error("❌ No training data provided!")
            return False
        
        print(f"\n🧠 Training model with {len(training_texts)} samples...")
        
        # Check class distribution
        intent_counts = Counter(training_intents)
        print(f"📊 Class distribution before balancing:")
        for intent, count in intent_counts.most_common():
            print(f"   • {intent}: {count} samples")
        
        # Balance the dataset by oversampling minority classes
        balanced_texts = []
        balanced_intents = []
        
        # Find target count (average of top 3 classes)
        sorted_counts = sorted(intent_counts.values(), reverse=True)
        target_count = int(np.mean(sorted_counts[:3]))
        
        for intent in intent_counts:
            # Get all samples for this intent
            intent_samples = [(text, intent_label) 
                             for text, intent_label in zip(training_texts, training_intents) 
                             if intent_label == intent]
            
            # Add original samples
            for text, intent_label in intent_samples:
                balanced_texts.append(text)
                balanced_intents.append(intent_label)
            
            # If this class has fewer samples, oversample it
            if len(intent_samples) < target_count:
                needed = target_count - len(intent_samples)
                for _ in range(needed):
                    text, intent_label = random.choice(intent_samples)
                    balanced_texts.append(text)
                    balanced_intents.append(intent_label)
        
        print(f"📊 After balancing: {len(balanced_texts)} samples")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            balanced_texts, balanced_intents, 
            test_size=0.2, random_state=42, 
            stratify=balanced_intents
        )
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Calculate accuracy
        train_predictions = self.pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        val_predictions = self.pipeline.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        print(f"✅ Model trained successfully!")
        print(f"📈 Total intents: {len(set(training_intents))}")
        print(f"📈 Intent classes: {sorted(set(training_intents))}")
        print(f"📈 Training accuracy: {train_accuracy:.2%}")
        print(f"📈 Validation accuracy: {val_accuracy:.2%}")
        
        # Show classification report for problematic intents
        problem_intents = ['financing', 'find_ready_property', 'process_info']
        problem_mask = [y in problem_intents for y in y_val]
        
        if any(problem_mask):
            X_val_problem = [X_val[i] for i in range(len(X_val)) if problem_mask[i]]
            y_val_problem = [y_val[i] for i in range(len(y_val)) if problem_mask[i]]
            
            if X_val_problem:
                val_problem_predictions = self.pipeline.predict(X_val_problem)
                print(f"\n🔍 Classification report for problem intents:")
                print(classification_report(y_val_problem, val_problem_predictions))
        
        # Show misclassified examples
        misclassified = []
        for i, (true, pred) in enumerate(zip(y_val, val_predictions)):
            if true != pred:
                misclassified.append({
                    'text': X_val[i],
                    'true': true,
                    'pred': pred
                })
        
        if misclassified:
            print(f"\n⚠️  Found {len(misclassified)} misclassified validation samples:")
            for i, case in enumerate(misclassified[:10]):
                display_text = case['text'][:60] + '...' if len(case['text']) > 60 else case['text']
                print(f"   {i+1}. '{display_text}'")
                print(f"       → True: {case['true']}, Pred: {case['pred']}")
        
        return True

    def save_model(self, model_path='models/nlu_model.pkl'):
        """Save trained model with version info"""
        model_data = {
            'vectorizer': self.pipeline.named_steps['tfidf'],
            'classifier': self.pipeline.named_steps['classifier'],
            'classes': self.pipeline.classes_.tolist(),
            'version': '3.4',  # Updated version
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.pipeline.named_steps['tfidf'].get_feature_names_out()),
            'intent_mapping': self.intent_mapping,
            'template_intent_map': self.template_intent_map,
            'intent_keywords': self.intent_keywords,
            'batangas_data_loaded': bool(self.batangas_data)
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to {model_path}")
        print(f"📊 Model info:")
        print(f"   • Version: {model_data['version']}")
        print(f"   • Classes: {len(model_data['classes'])} intents")
        print(f"   • Date: {model_data['training_date']}")
        print(f"   • Features: {model_data['feature_count']}")
        print(f"   • Batangas Data: {'✅ Loaded' if model_data['batangas_data_loaded'] else '❌ Not loaded'}")
        
        return model_path

def test_predictions(trainer, test_queries):
    """Test model predictions with the specific problematic queries"""
    print("\n" + "="*60)
    print("🧪 TESTING PROBLEMATIC QUERIES")
    print("="*60)
    
    # Test the specific queries that were misclassified
    specific_queries = [
        "Properties that accept bank financing",
        "Find ready to move in properties for students in Batangas City",
        "Steps for buying a condo",
        "properties with swimming pool",
        "properties near schools",
        "how to get mortgage",
        "tell me about lipa city",
        "available now apartments",
        "houses for big family",
        "condos near malls"
    ]
    
    for query in specific_queries:
        try:
            intent = trainer.pipeline.predict([query])[0]
            proba = trainer.pipeline.predict_proba([query])[0]
            confidence = max(proba) * 100
            intent_idx = list(trainer.pipeline.classes_).index(intent)
            
            print(f"🔍 '{query}'")
            print(f"   → Intent: {intent} ({confidence:.1f}% confidence)")
            
            # Show top 3 intents for ambiguous queries
            if confidence < 80:
                top_indices = np.argsort(proba)[-3:][::-1]
                print(f"   Top alternatives:")
                for idx in top_indices:
                    if idx != intent_idx:
                        intent_name = trainer.pipeline.classes_[idx]
                        intent_prob = proba[idx] * 100
                        if intent_prob > 10:  # Only show significant alternatives
                            print(f"     • {intent_name}: {intent_prob:.1f}%")
            print()
        except Exception as e:
            print(f"❌ Error predicting '{query}': {e}")

def create_additional_training_file():
    """Create/update additional training data file"""
    additional_data = {
        "additional_samples": [
            {"text": "hi", "intent": "greeting"},
            {"text": "hello", "intent": "greeting"},
            {"text": "hey", "intent": "greeting"},
                        # Force "find [property] in [location]" to be find_property
            {"text": "find apartments in batangas city", "intent": "find_property"},
            {"text": "find apartment in batangas city", "intent": "find_property"},
            {"text": "find apartments in batangas", "intent": "find_property"},
            {"text": "search apartments in batangas city", "intent": "find_property"},
            {"text": "look for apartments in batangas city", "intent": "find_property"},
            {"text": "show me apartments in batangas city", "intent": "find_property"},
            
            # Make "tell me about" clearly location_info
            {"text": "tell me about batangas city apartments", "intent": "location_info"},
            {"text": "information about apartments in batangas", "intent": "location_info"},
            {"text": "what is batangas city like for apartments", "intent": "location_info"},
            
            # Clear distinction between the two patterns:
            # Pattern 1: "find X in Y" = find_property
            {"text": "find house in lipa", "intent": "find_property"},
            {"text": "search house in lipa city", "intent": "find_property"},
            {"text": "look for house in lipa", "intent": "find_property"},
            {"text": "show me houses in lipa city", "intent": "find_property"},
            
            # Pattern 2: "tell me about Y" = location_info
            {"text": "tell me about lipa city houses", "intent": "location_info"},
            {"text": "what is lipa city like for houses", "intent": "location_info"},
            {"text": "information about houses in lipa", "intent": "location_info"},
            
            # Negative examples - what should NOT be location_info
            {"text": "find apartments in the city", "intent": "find_property"},
            {"text": "search for properties in that city", "intent": "find_property"},
            {"text": "looking for houses in the city", "intent": "find_property"},
            
            # Additional problematic patterns from your logs
            {"text": "search property nasugbu", "intent": "find_property"},
            {"text": "condo living in tanauan", "intent": "find_property"},
            
             # General property searches (no location)
            {"text": "find apartments", "intent": "find_property"},
            {"text": "show me houses", "intent": "find_property"},
            {"text": "look for condos", "intent": "find_property"},
            {"text": "search for properties", "intent": "find_property"},
            {"text": "i need a house", "intent": "find_property"},
            {"text": "show me available properties", "intent": "find_property"},
            {"text": "find beachfront properties", "intent": "find_property"},
            {"text": "what apartments do you have", "intent": "find_property"},
            {"text": "what houses are available", "intent": "find_property"},
            {"text": "show me all condos", "intent": "find_property"},
            {"text": "do you have any apartments", "intent": "find_property"},
            {"text": "any houses for sale", "intent": "find_property"},
            {"text": "any condos for rent", "intent": "find_property"},
            {"text": "properties for rent", "intent": "find_property"},
            {"text": "properties for sale", "intent": "find_property"},
            
            # Force "find X in Y" to be find_property (not location_info)
             {"text": "show me house in nasugbu", "intent": "find_property"},
            {"text": "show me houses in nasugbu", "intent": "find_property"},
            {"text": "show me properties in nasugbu", "intent": "find_property"},
            {"text": "show me apartments in batangas city", "intent": "find_property"},
            {"text": "show me condos in lipa city", "intent": "find_property"},
            {"text": "show me beachfront properties in nasugbu", "intent": "find_property"},
            
            # Make "show me" more distinct from location_info
            {"text": "show me what's available in nasugbu", "intent": "find_property"},
            {"text": "show me options in nasugbu", "intent": "find_property"},
            {"text": "show me listings in nasugbu", "intent": "find_property"},
            {"text": "find apartments in batangas city", "intent": "find_property"},
            {"text": "find house in nasugbu", "intent": "find_property"},
            {"text": "find condos in lipa city", "intent": "find_property"},
            {"text": "find townhouses in sto. tomas", "intent": "find_property"},
            {"text": "find commercial spaces in batangas city", "intent": "find_property"},
            {"text": "find beachfront properties in nasugbu", "intent": "find_property"},
            {"text": "find resort properties in nasugbu", "intent": "find_property"},
            {"text": "find agricultural land in nasugbu", "intent": "find_property"},
            {"text": "find apartments in nasugbu", "intent": "find_property"},
            
            # More variations with "find" keyword
            {"text": "look for apartments in batangas city", "intent": "find_property"},
            {"text": "search for apartments in batangas", "intent": "find_property"},
            {"text": "i need apartments in batangas city", "intent": "find_property"},
            {"text": "show me houses in batangas city", "intent": "find_property"},
            {"text": "show me condos in tanauan city", "intent": "find_property"},
            {"text": "i'm looking for a house in lipa", "intent": "find_property"},
            {"text": "can you find apartments in batangas city", "intent": "find_property"},
            {"text": "please locate apartments in batangas", "intent": "find_property"},
            
            # Location info samples should NOT have "find" keyword
            {"text": "tell me about batangas city", "intent": "location_info"},
            {"text": "what is batangas city like", "intent": "location_info"},
            {"text": "describe batangas city", "intent": "location_info"},
            {"text": "living in batangas city", "intent": "location_info"},
            {"text": "about batangas city", "intent": "location_info"},
            {"text": "information about batangas city", "intent": "location_info"},
            {"text": "what's it like in batangas city", "intent": "location_info"},
            {"text": "batangas city lifestyle", "intent": "location_info"},
            {"text": "how is life in batangas city", "intent": "location_info"},
            
            # ============================================
            # ORIGINAL SAMPLES (with improvements)
            # ============================================
            
            # Financing samples - with more variations
            {"text": "properties that accept bank financing", "intent": "financing"},
            {"text": "houses that accept bank loans", "intent": "financing"},
            {"text": "how to get bank financing", "intent": "financing"},
            {"text": "bank financing requirements", "intent": "financing"},
            {"text": "pag-ibig financing requirements", "intent": "financing"},
            {"text": "properties with in-house financing", "intent": "financing"},
            {"text": "properties that accept cash payment", "intent": "financing"},
            {"text": "properties that accept installment payment", "intent": "financing"},
            {"text": "what documents are needed for bank financing", "intent": "financing"},
            
            # Ready property samples - more specific
            {"text": "ready to move in properties", "intent": "find_ready_property"},
            {"text": "available now properties", "intent": "find_ready_property"},
            {"text": "immediate occupancy houses", "intent": "find_ready_property"},
            {"text": "move in ready condos", "intent": "find_ready_property"},
            {"text": "ready for occupancy apartments", "intent": "find_ready_property"},
            {"text": "ready to occupy units", "intent": "find_ready_property"},
            {"text": "properties available immediately", "intent": "find_ready_property"},
            {"text": "find ready to move in properties", "intent": "find_ready_property"},
            
            # Process info samples
            {"text": "steps for buying", "intent": "process_info"},
            {"text": "how to buy a property", "intent": "process_info"},
            {"text": "property purchase process", "intent": "process_info"},
            {"text": "timeline for buying a house", "intent": "process_info"},
            {"text": "requirements for property purchase", "intent": "process_info"},
            {"text": "steps for buying a condo", "intent": "process_info"},
            {"text": "how to get a mortgage", "intent": "process_info"},
            {"text": "process for renting", "intent": "process_info"},
            
            # With feature samples
            {"text": "properties with swimming pool", "intent": "find_with_feature"},
            {"text": "houses with garden", "intent": "find_with_feature"},
            {"text": "apartments with parking", "intent": "find_with_feature"},
            {"text": "condos with security", "intent": "find_with_feature"},
            {"text": "properties with wifi", "intent": "find_with_feature"},
            {"text": "find properties with swimming pool", "intent": "find_with_feature"},
            {"text": "show me houses with garden", "intent": "find_with_feature"},
            {"text": "look for apartments with parking", "intent": "find_with_feature"},
            
            # Near landmark samples
            {"text": "properties near schools", "intent": "find_near_landmark"},
            {"text": "houses near malls", "intent": "find_near_landmark"},
            {"text": "apartments near hospitals", "intent": "find_near_landmark"},
            {"text": "condos near beaches", "intent": "find_near_landmark"},
            {"text": "properties near churches", "intent": "find_near_landmark"},
            {"text": "find properties near schools", "intent": "find_near_landmark"},
            {"text": "show me houses near malls", "intent": "find_near_landmark"},
            {"text": "look for apartments near hospitals", "intent": "find_near_landmark"},
            
            # More location info samples
            {"text": "tell me about lipa city", "intent": "location_info"},
            {"text": "what is lipa city like", "intent": "location_info"},
            {"text": "describe tanauan city", "intent": "location_info"},
            {"text": "information about nasugbu", "intent": "location_info"},
            {"text": "living in san juan", "intent": "location_info"},
            {"text": "tell me about calatagan", "intent": "location_info"},
            {"text": "what's it like to live in taal", "intent": "location_info"},
            {"text": "describe mabini batangas", "intent": "location_info"},
            {"text": "information about sto. tomas city", "intent": "location_info"},
            
            # ============================================
            # ADDITIONAL CONTEXTUAL SAMPLES
            # ============================================
            
            # Clear distinction samples
            {"text": "i want to find a house in batangas", "intent": "find_property"},
            {"text": "i want to know about batangas city", "intent": "location_info"},
            {"text": "search properties in nasugbu", "intent": "find_property"},
            {"text": "give me information about nasugbu", "intent": "location_info"},
            {"text": "show me available properties in lipa", "intent": "find_property"},
            {"text": "tell me about living in lipa", "intent": "location_info"},
            
            # Property for need samples
            {"text": "properties for family needs in lipa", "intent": "find_property_for_need"},
            {"text": "houses for big family", "intent": "find_property_for_need"},
            {"text": "apartments for students in batangas city", "intent": "find_property_for_need"},
            {"text": "condos for professionals", "intent": "find_property_for_need"},
            {"text": "properties for retirees", "intent": "find_property_for_need"},
            
            # Match needs samples
            {"text": "match properties to my budget", "intent": "match_needs"},
            {"text": "recommend properties for me", "intent": "match_needs"},
            {"text": "find suitable properties", "intent": "match_needs"},
            {"text": "what properties match my needs", "intent": "match_needs"},
            
            # Property with criteria samples
            {"text": "houses under 3M with 3 bedrooms", "intent": "find_property_with_criteria"},
            {"text": "apartments under 15000 pesos", "intent": "find_property_with_criteria"},
            {"text": "condos with 2 bedrooms and 2 bathrooms", "intent": "find_property_with_criteria"},
            {"text": "properties under 10M with swimming pool", "intent": "find_property_with_criteria"},
            
            # ============================================
            # NEGATIVE EXAMPLES (what should NOT match)
            # ============================================
            
            # These should NOT be location_info (they have "find" keywords)
            {"text": "find tell me about batangas city", "intent": "find_property"},  # Edge case
            {"text": "search information about lipa", "intent": "find_property"},     # Mixed intent
            
            # These should NOT be find_property (they don't have search keywords)
            {"text": "about finding houses", "intent": "location_info"},  # Has "finding" but starts with "about"
            {"text": "information on searching properties", "intent": "location_info"},  # Has "searching" but starts with "information"
        ]
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/additional_training.json', 'w', encoding='utf-8') as f:
        json.dump(additional_data, f, indent=2)
    
    print("✅ Created/updated additional_training.json with specific intent corrections")
    print(f"   Total samples: {len(additional_data['additional_samples'])}")
    print(f"   Focus: Fixing 'find X in Y' vs 'tell me about Y' confusion")
    
def main():
    print("="*60)
    print("🚀 BAH.AI PROPERTY CHATBOT TRAINING SYSTEM v3.4")
    print("   (With intent classification fixes)")
    print("="*60)
    
    # Create/update additional training data file
    create_additional_training_file()
    
    # Initialize trainer
    trainer = TeamNLUTrainer()
    
    # Load and train using ALL data sources
    texts, intents = trainer.load_all_training_data('data')
    
    if texts:
        if trainer.train(texts, intents):
            trainer.save_model()
            
            # Test with the specific problematic queries
            test_predictions(trainer, [
                "Properties that accept bank financing",
                "Find ready to move in properties for students in Batangas City",
                "Steps for buying a condo",
                "properties with swimming pool",
                "properties near schools",
                "how to get mortgage",
                "tell me about lipa city",
                "available now apartments",
                "houses for big family",
                "condos near malls"
            ])
            
            # Also test with some general queries
            print("\n" + "="*60)
            print("🧪 TESTING GENERAL QUERIES")
            print("="*60)
            
            general_queries = [
                "find apartments in batangas city",
                "show me houses under 3M with 3 bedrooms",
                "properties for family needs in lipa",
                "ready to move in condos",
                "how to get a pag-ibig loan",
                "tell me about tanauan city",
                "properties with garden at reasonable cost",
                "match properties to my budget as single professional",
                "houses under 10M with swimming pool",
                "what documents are needed for bank financing"
            ]
            
            for query in general_queries:
                try:
                    intent = trainer.pipeline.predict([query])[0]
                    proba = trainer.pipeline.predict_proba([query])[0]
                    confidence = max(proba) * 100
                    print(f"🔍 '{query}'")
                    print(f"   → Intent: {intent} ({confidence:.1f}% confidence)")
                except Exception as e:
                    print(f"❌ Error predicting '{query}': {e}")
    else:
        print("❌ No training data found!")
        print("💡 Make sure your data folder structure is:")
        print("   data/")
        print("   ├── member1/training_data.json")
        print("   ├── member2/training_data.json")
        print("   ├── member3/training_data.json")
        print("   ├── additional_training.json")
        print("   └── shared/")
        print("       ├── all_questions.json")
        print("       ├── synonyms.json")
        print("       └── batangas_complete.json")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()