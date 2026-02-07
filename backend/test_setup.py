# backend/test_new_key.py
import os
import firebase_admin
from firebase_admin import credentials, firestore

print("=" * 60)
print("🔑 TESTING NEW SERVICE ACCOUNT KEY")
print("=" * 60)

# Get the correct path
current_dir = os.path.dirname(os.path.abspath(__file__))  # backend
root_dir = os.path.dirname(current_dir)                   # root
key_path = os.path.join(root_dir, 'serviceAccountKey.json')

print(f"📁 Looking for key at: {key_path}")
print(f"✅ File exists: {os.path.exists(key_path)}")

if os.path.exists(key_path):
    try:
        # 1. Load the credential
        print("\n1️⃣ Loading credential...")
        cred = credentials.Certificate(key_path)
        print("✅ Credential loaded")
        
        # 2. Initialize Firebase
        print("2️⃣ Initializing Firebase...")
        app = firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized")
        
        # 3. Test Firestore
        print("3️⃣ Testing Firestore...")
        db = firestore.client()
        print("✅ Firestore client created")
        
        # 4. Try to list collections (simple operation)
        print("4️⃣ Listing collections...")
        collections = db.collections()
        count = 0
        for collection in collections:
            print(f"   - {collection.id}")
            count += 1
        
        print(f"\n✅ SUCCESS! Found {count} collections")
        
        # 5. Try to get properties
        print("\n5️⃣ Checking properties collection...")
        try:
            docs = db.collection('properties').limit(3).get()
            prop_count = len(docs)
            print(f"✅ Found {prop_count} properties")
            
            if prop_count > 0:
                print("\n📋 Sample properties:")
                for i, doc in enumerate(docs):
                    data = doc.to_dict()
                    print(f"\n  {i+1}. {doc.id}")
                    print(f"     Title: {data.get('title', 'N/A')}")
                    print(f"     Type: {data.get('propertyType', data.get('type', 'N/A'))}")
                    print(f"     City: {data.get('city', 'N/A')}")
                    print(f"     Status: {data.get('status', 'N/A')}")
                    
        except Exception as e:
            print(f"⚠️ No properties found or error: {e}")
            print("💡 This is OK if you haven't added properties yet")
        
        print("\n" + "=" * 60)
        print("🎉 FIREBASE IS WORKING CORRECTLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"\n❌ Key file not found!")
    print("💡 Make sure serviceAccountKey.json is in:")
    print(f"   {root_dir}")