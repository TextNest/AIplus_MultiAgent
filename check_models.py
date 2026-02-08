import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("GOOGLE_API_KEY not found.")
else:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("Listing available models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Name: {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
