import json
import re

def fix_json_file(filepath):
    """Fix common JSON syntax errors"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove trailing commas (most common issue)
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)

    # Write back the cleaned content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    # Verify JSON validity
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ JSON fixed and loaded successfully")

        if isinstance(data, dict) and "training_samples" in data:
            print(f"📦 Samples found: {len(data['training_samples'])}")
        return True

    except json.JSONDecodeError as e:
        print("❌ JSON still has an error")
        print(f"Error: {e}")
        print(f"Line: {e.lineno}, Column: {e.colno}")
        return False


# RUN THE FIX
fix_json_file("backend/data/member1/training_data.json")
