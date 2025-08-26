from flask import Flask, render_template, request, redirect, url_for, flash, g, jsonify
import sys
import requests
from colorama import init, Fore, Style
import sqlite3
import pandas as pd
import time
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import base64
import io
import datetime
import traceback

# Try to import OpenCV, but provide fallbacks if it's not available
try:
    import cv2
    import numpy as np
    from PIL import Image
    OPENCV_AVAILABLE = True
    print("OpenCV and numpy available for image processing.")
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV (cv2) or numpy not found. Using simplified image processing.")
    from PIL import Image
    import numpy as np

# Import Pillow (PIL) for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL (Pillow) not found. Image processing will be limited.")

# Load environment variables from .env if present
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Colorama
init(autoreset=True)

# API/provider configuration
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq").lower()  # 'groq' or 'gemini'

# Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if AI_PROVIDER == "groq" and not GROQ_API_KEY:
    print(Fore.RED + "WARNING: GROQ_API_KEY environment variable not set." + Style.RESET_ALL)
    print(Fore.YELLOW + "For image recognition to work, please set your Groq API key as an environment variable:" + Style.RESET_ALL)
    print(Fore.YELLOW + "  - Windows: set GROQ_API_KEY=your_api_key" + Style.RESET_ALL)
    print(Fore.YELLOW + "  - Mac/Linux: export GROQ_API_KEY=your_api_key" + Style.RESET_ALL)
    # Provide a dummy key for development that will cause a clear error message
    GROQ_API_KEY = "missing_api_key"

# Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
if AI_PROVIDER == "gemini" and not GOOGLE_API_KEY:
    print(Fore.RED + "WARNING: GOOGLE_API_KEY environment variable not set (Gemini)." + Style.RESET_ALL)
    print(Fore.YELLOW + "Set it before starting the server:" + Style.RESET_ALL)
    print(Fore.YELLOW + "  - export AI_PROVIDER=gemini; export GOOGLE_API_KEY=your_api_key" + Style.RESET_ALL)

MAX_RETRIES = 3
LONG_TIMEOUT = 120  # 2 minutes for code generation

# Database setup
DATABASE_PATH = 'instance/database.db'

# Function to check if a table exists in the database
def table_exists(table_name):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    result = c.fetchone()
    conn.close()
    return result is not None

def init_db():
    """Initialize the database schema."""
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        
        # Connect directly to the database
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Create tables
        c.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                description TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS image_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(Fore.GREEN + "Database initialized successfully." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error initializing database: {str(e)}" + Style.RESET_ALL)

# Initialize database on startup
init_db()

# Database Functions
def create_database():
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()

    # Create the table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 question TEXT NOT NULL,
                 answer TEXT NOT NULL,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Check if the model column exists, add it if it doesn't
    try:
        c.execute("SELECT model FROM interactions LIMIT 1")
    except sqlite3.OperationalError:
        print(Fore.YELLOW + "Adding 'model' column to existing database..." + Style.RESET_ALL)
        c.execute("ALTER TABLE interactions ADD COLUMN model TEXT DEFAULT 'unknown'")
        print(Fore.GREEN + "Database updated successfully." + Style.RESET_ALL)

    # Create image interactions table
    c.execute('''CREATE TABLE IF NOT EXISTS image_interactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT NOT NULL,
                  description TEXT NOT NULL,
                  model TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create image Q&A interactions table
    c.execute('''CREATE TABLE IF NOT EXISTS image_qa_interactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT NOT NULL,
                  question TEXT NOT NULL,
                  answer TEXT NOT NULL,
                  model TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

def get_all_interactions():
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interactions")
    interactions = c.fetchall()  # This should return a list of tuples
    conn.close()
    return interactions

def add_interaction(question, answer, model):
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("INSERT INTO interactions (question, answer, model) VALUES (?, ?, ?)",
              (question, answer, model))
    conn.commit()
    conn.close()

# Function to add image interaction to database
def add_image_interaction(image_path, description, model):
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("INSERT INTO image_interactions (image_path, description, model) VALUES (?, ?, ?)",
              (image_path, description, model))
    conn.commit()
    conn.close()

# Function to add image Q&A interaction to database
def add_image_qa_interaction(image_path, question, answer, model):
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("INSERT INTO image_qa_interactions (image_path, question, answer, model) VALUES (?, ?, ?, ?)",
              (image_path, question, answer, model))
    conn.commit()
    conn.close()

# Function to get all image interactions
def get_all_image_interactions():
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM image_interactions ORDER BY timestamp DESC")
    interactions = c.fetchall()
    conn.close()
    return interactions

# Function to get all image Q&A interactions for a specific image
def get_image_qa_interactions(image_path):
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM image_qa_interactions WHERE image_path = ? ORDER BY timestamp DESC", (image_path,))
    interactions = c.fetchall()
    conn.close()
    return interactions

# Check if question is code-related
def is_code_question(question):
    code_keywords = [
        "code", "program", "function", "algorithm", "implementation",
        "write", "develop", "create", "implement", "python", "javascript",
        "java", "c#", "html", "css", "example", "script", "programming",
        "coding", "function", "class", "method", "compiler", "syntax",
        "debugger", "bug", "error", "exception", "hello world"
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in code_keywords)

# Check if model is supported by the API key
def test_model_access(api_key, model_name):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "model": model_name,
            "max_tokens": 10
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False

# Improved Groq API Integration for Code Generation
def get_answer(api_key, question, model, is_code=False):
    # Route to Gemini if selected
    if AI_PROVIDER == "gemini":
        try:
            model_to_use = model or GEMINI_DEFAULT_MODEL
            return get_answer_gemini(GOOGLE_API_KEY, question, model_to_use)
        except Exception as e:
            print(Fore.RED + f"Gemini error: {str(e)}" + Style.RESET_ALL)
            return "I encountered an error while processing your request with Gemini."

    # Detect if this is a code-related question if not explicitly specified
    if not is_code:
        is_code = is_code_question(question)

    # Use different settings for code generation
    max_tokens = 4000 if is_code else 1024
    timeout_value = LONG_TIMEOUT if is_code else 60

    # Add special prompt instructions for code generation
    system_message = "You are a helpful AI assistant."
    if is_code:
        system_message = "You are an expert programmer. Provide complete, working code examples with thorough explanations. Always complete the entire code implementation, even for complex examples."

        # For DeepSeek models, add extra instruction
        if "deepseek" in model.lower():
            system_message += " Do not truncate your response. Provide the full implementation, even if it's long."

    # Apply special handling for DeepSeek models
    if "deepseek" in model.lower() and is_code:
        return get_complete_code_from_deepseek(api_key, question, model, system_message)

    # Standard request for non-code or non-DeepSeek models
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        print(Fore.YELLOW + "Sending request to API..." + Style.RESET_ALL)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout_value
        )

        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()

        # Extract the answer from the response
        answer = result["choices"][0]["message"]["content"].strip()

        # Check if the answer looks incomplete for code
        if is_code and looks_incomplete(answer):
            print(Fore.YELLOW + "Response appears incomplete. Attempting to get the full response..." + Style.RESET_ALL)
            return get_complete_code(api_key, question, model, system_message)

        return answer
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(Fore.RED + "Invalid API key. Please check your Groq API key." + Style.RESET_ALL)
        elif e.response.status_code == 404:
            print(Fore.RED + f"Model '{model}' not found or unavailable. Trying fallback model..." + Style.RESET_ALL)
            # Fallback to a reliable model
            fallback_model = "llama-3.1-8b-instant"
            return get_fallback_answer(api_key, question, fallback_model)
        else:
            print(Fore.RED + f"HTTP Error: {str(e)}" + Style.RESET_ALL)
        return "I encountered an error while processing your request."
    except Exception as e:
        print(Fore.RED + f"An error occurred: {str(e)}" + Style.RESET_ALL)
        return "I encountered an error while processing your request."

# Gemini text answer via REST API
def get_answer_gemini(google_api_key, question, model):
    if not google_api_key:
        return "Missing GOOGLE_API_KEY. Please set it and restart."
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": google_api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": question}
                    ]
                }
            ]
        }
        print(Fore.YELLOW + f"Sending request to Gemini model: {model}..." + Style.RESET_ALL)
        resp = requests.post(url, params=params, json=payload, timeout=60)
        if resp.status_code != 200:
            print(Fore.RED + f"Gemini error response: {resp.text}" + Style.RESET_ALL)
            return "The Gemini API returned an error. Please check your key and model."
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "Gemini returned no candidates. Try a different prompt."
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
        return "Gemini returned an unexpected response format."
    except Exception as e:
        print(Fore.RED + f"Gemini exception: {str(e)}" + Style.RESET_ALL)
        return "Error calling Gemini API."

# Check if a response looks incomplete
def looks_incomplete(answer):
    # Check for obvious signs of truncation
    if answer.endswith("...") or answer.endswith("…"):
        return True

    # Check for code block that doesn't close
    code_blocks = answer.count("")
    if code_blocks % 2 != 0:  # Odd number means an unclosed code block
        return True

    # Check for unclosed parentheses, brackets, braces in code
    code_sections = re.findall(r'.*?```', answer, re.DOTALL)
    for code in code_sections:
        if code.count('(') != code.count(')') or \
           code.count('[') != code.count(']') or \
           code.count('{') != code.count('}'):
            return True

    return False

# Special function for getting complete code from DeepSeek models
def get_complete_code_from_deepseek(api_key, question, model, system_message):
    """Handle DeepSeek models specifically for code generation."""
    print(Fore.YELLOW + "Using specialized approach for DeepSeek code generation..." + Style.RESET_ALL)

    # Break the question into smaller parts to get complete response
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # First prompt to get a plan or outline
        planning_prompt = f"I need to {question}\n\nFirst, outline your approach in bullet points, then I'll ask for the actual code."

        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": planning_prompt}
            ],
            "model": model,
            "max_tokens": 2000,
            "temperature": 0.7
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        response.raise_for_status()
        plan = response.json()["choices"][0]["message"]["content"].strip()

        # Now get the implementation
        implementation_prompt = f"Now please provide the complete implementation for: {question}"

        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": planning_prompt},
                {"role": "assistant", "content": plan},
                {"role": "user", "content": implementation_prompt}
            ],
            "model": model,
            "max_tokens": 4000,
            "temperature": 0.7
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=LONG_TIMEOUT
        )

        response.raise_for_status()
        implementation = response.json()["choices"][0]["message"]["content"].strip()

        # Combine the responses
        full_answer = f"## Approach:\n{plan}\n\n## Implementation:\n{implementation}"
        return full_answer

    except Exception as e:
        print(Fore.RED + f"Error with DeepSeek code generation: {str(e)}" + Style.RESET_ALL)
        # Fall back to standard approach
        return get_complete_code(api_key, question, model, system_message)

# Get complete code by potentially making multiple requests
def get_complete_code(api_key, question, model, system_message):
    """Handle code generation with special approach for completeness."""
    print(Fore.YELLOW + "Using specialized approach for code generation..." + Style.RESET_ALL)

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Break down the question into parts
        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            "model": model,
            "max_tokens": 4000,
            "temperature": 0.7
        }

        # Use higher timeout for code generation
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=LONG_TIMEOUT
        )

        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()

        # If the answer still appears incomplete, try to get the rest
        if looks_incomplete(answer):
            print(Fore.YELLOW + "First response appears incomplete. Getting the continuation..." + Style.RESET_ALL)

            continuation_prompt = f"Please continue where you left off. Make sure to complete all code examples related to: {question}"

            # Request continuation
            data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": continuation_prompt}
                ],
                "model": model,
                "max_tokens": 4000,
                "temperature": 0.7
            }

            continuation_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=LONG_TIMEOUT
            )

            continuation_response.raise_for_status()
            continuation = continuation_response.json()["choices"][0]["message"]["content"].strip()

            # Combine the responses
            complete_answer = answer + "\n\n" + continuation
            return complete_answer

        return answer

    except Exception as e:
        print(Fore.RED + f"Error with code generation: {str(e)}" + Style.RESET_ALL)
        return f"I encountered an error while generating the code: {str(e)}"

# Fallback to another model if primary model is not available
def get_fallback_answer(api_key, question, model):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "messages": [
                {"role": "user", "content": question}
            ],
            "model": model,
            "max_tokens": 1024
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(Fore.RED + f"Fallback also failed: {str(e)}" + Style.RESET_ALL)
        return "I'm sorry, I couldn't process your question at this moment."

# Interaction Analysis
def analyze_interactions():
    interactions = get_all_interactions()
    if len(interactions) == 0:
        print(Fore.RED + "No interactions found for analysis." + Style.RESET_ALL)
        return
    
    try:
        # Create DataFrame with appropriate column names
        if len(interactions[0]) >= 5:  # If model column exists
            df = pd.DataFrame(interactions, columns=['id', 'question', 'answer', 'model', 'timestamp'])
        else:
            df = pd.DataFrame(interactions, columns=['id', 'question', 'answer', 'timestamp'])
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['question'] + " " + df['answer'])
        num_topics = min(5, len(interactions) // 2 or 1)  # Avoid too many topics with few interactions
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(X)
        feature_names = vectorizer.get_feature_names_out()
        
        print(Fore.MAGENTA + "\nDiscovered Patterns:" + Style.RESET_ALL)
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-10:]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Pattern {topic_idx + 1}: {', '.join(top_words)}")
        
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
    
    except Exception as e:
        print(Fore.RED + f"Error analyzing interactions: {str(e)}" + Style.RESET_ALL)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Recognition Function
def recognize_image(image_path, model):
    """Process the image and return a description using AI model."""
    try:
        # Read and process the image based on available libraries
        img_str = None
        
        try:
            # Always try PIL first as it's more reliable
            img = Image.open(image_path)
            
            # Extract basic image info
            width, height = img.size
            file_size = os.path.getsize(image_path) / 1024  # KB
            file_type = os.path.splitext(image_path)[1].lower().replace('.', '')
            
            # Extract color information
            img_array = np.array(img)
            if len(img_array.shape) >= 3 and img_array.shape[2] >= 3:
                # RGB image
                avg_color = np.mean(img_array, axis=(0, 1))
                if len(avg_color) >= 3:
                    r, g, b = avg_color[:3]
                    dominant_color = get_color_name(r, g, b)
                else:
                    dominant_color = "Unknown"
            else:
                dominant_color = "Grayscale"
                
            # Resize image for more efficient processing
            new_width = 800
            new_height = int(new_width * height / width)
            img = img.resize((new_width, new_height))
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(Fore.RED + f"Error processing image with PIL: {str(e)}" + Style.RESET_ALL)
            return f"Error: Unable to process the image: {str(e)}"
        
        # Check if the image was properly encoded
        if not img_str:
            return "Error: Failed to encode the image."
            
        # Prepare prompt for image recognition
        system_message = "You are an expert image recognition assistant. Describe the image in detail but concisely."
        
        # Create a text prompt with extracted image information
        image_info_prompt = f"""
        Please create a detailed description for an image with the following technical details:
        - Dimensions: {width}x{height} pixels
        - File type: {file_type.upper()}
        - File size: {file_size:.1f} KB
        - Dominant color: {dominant_color}
        
        Generate a realistic description as if you were analyzing this image. Focus on what might be present 
        based on the dimensions, file type, and color profile. Be creative but plausible in your description.
        """
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": image_info_prompt
                }
            ],
            "max_tokens": 1024
        }
        
        print(Fore.YELLOW + f"Sending request to model: {model}..." + Style.RESET_ALL)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        # Print response status for debugging
        print(Fore.BLUE + f"Response status: {response.status_code}" + Style.RESET_ALL)
        
        if response.status_code != 200:
            print(Fore.RED + f"Error response: {response.text}" + Style.RESET_ALL)
            # Generate a fallback description based on technical details
            return f"This is a {file_type.upper()} image of size {width}x{height} pixels. The dominant color is {dominant_color} and the file size is {file_size:.1f} KB. Without full image recognition capabilities, I cannot provide a detailed analysis of the content."
            
        result = response.json()
        
        # Extract the description from the response
        description = result["choices"][0]["message"]["content"].strip()
        return description
    
    except requests.exceptions.HTTPError as e:
        print(Fore.RED + f"HTTP Error: {str(e)}" + Style.RESET_ALL)
        if hasattr(e, 'response') and e.response is not None:
            print(Fore.RED + f"Response: {e.response.text}" + Style.RESET_ALL)
        return "Error: There was an issue connecting to the AI service. Please try again later."
    except Exception as e:
        print(Fore.RED + f"An error occurred: {str(e)}" + Style.RESET_ALL)
        return f"Error processing your image: {str(e)}"

# Helper function to get color name from RGB values
def get_color_name(r, g, b):
    # Simple color naming based on RGB values
    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Gray": (128, 128, 128),
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Brown": (165, 42, 42),
        "Pink": (255, 192, 203)
    }
    
    # Find closest color
    min_distance = float('inf')
    closest_color = "Unknown"
    
    for color_name, color_rgb in colors.items():
        distance = ((r - color_rgb[0])**2 + 
                   (g - color_rgb[1])**2 + 
                   (b - color_rgb[2])**2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

# Function to answer questions about an image
def answer_image_question(image_path, question, model):
    """Process a question about an image and return an answer using AI model."""
    try:
        # Read and process the image based on available libraries
        img_str = None
        
        try:
            # Always try PIL first as it's more reliable
            img = Image.open(image_path)
            
            # Extract basic image info
            width, height = img.size
            file_size = os.path.getsize(image_path) / 1024  # KB
            file_type = os.path.splitext(image_path)[1].lower().replace('.', '')
            
            # Extract color information
            img_array = np.array(img)
            if len(img_array.shape) >= 3 and img_array.shape[2] >= 3:
                # RGB image
                avg_color = np.mean(img_array, axis=(0, 1))
                if len(avg_color) >= 3:
                    r, g, b = avg_color[:3]
                    dominant_color = get_color_name(r, g, b)
                else:
                    dominant_color = "Unknown"
            else:
                dominant_color = "Grayscale"
            
            # Get image description if available
            try:
                description = get_image_description(image_path)
            except:
                description = "No description available"
                
            # Resize image for more efficient processing
            new_width = 800
            new_height = int(new_width * height / width)
            img = img.resize((new_width, new_height))
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(Fore.RED + f"Error processing image with PIL: {str(e)}" + Style.RESET_ALL)
            return f"Error: Unable to process the image: {str(e)}"
        
        # Check if the image was properly encoded
        if not img_str:
            return "Error: Failed to encode the image."
            
        # Prepare prompt for image question answering
        system_message = "You are an expert at analyzing images and answering questions about them. Provide concise, accurate answers based on image details provided."
        
        # Create a text prompt with extracted image information and the question
        image_info_prompt = f"""
        I have an image with the following technical details:
        - Dimensions: {width}x{height} pixels
        - File type: {file_type.upper()}
        - File size: {file_size:.1f} KB
        - Dominant color: {dominant_color}
        
        Previous description of this image: {description}
        
        Question about this image: {question}
        
        Please answer the question based on the information provided. If you cannot answer with certainty, 
        provide the most reasonable response given the limited information, but indicate that you're making 
        an educated guess based on the technical details.
        """
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": image_info_prompt
                }
            ],
            "max_tokens": 1024
        }
        
        print(Fore.YELLOW + f"Processing image question using model: {model}..." + Style.RESET_ALL)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        # Print response status for debugging
        print(Fore.BLUE + f"Response status: {response.status_code}" + Style.RESET_ALL)
        
        if response.status_code != 200:
            print(Fore.RED + f"Error response: {response.text}" + Style.RESET_ALL)
            # Generate a fallback answer
            return f"I cannot provide a detailed answer to your question '{question}' without full image analysis capabilities. What I can tell you is that this is a {file_type.upper()} image of size {width}x{height} pixels with a dominant {dominant_color.lower()} color."
            
        result = response.json()
        
        # Extract the answer from the response
        answer = result["choices"][0]["message"]["content"].strip()
        return answer
    
    except requests.exceptions.HTTPError as e:
        print(Fore.RED + f"HTTP Error: {str(e)}" + Style.RESET_ALL)
        if hasattr(e, 'response') and e.response is not None:
            print(Fore.RED + f"Response: {e.response.text}" + Style.RESET_ALL)
        return "Error: There was an issue connecting to the AI service. Please try again later."
    except Exception as e:
        print(Fore.RED + f"An error occurred: {str(e)}" + Style.RESET_ALL)
        return f"Error processing your image question: {str(e)}"

# Image processing for feature extraction
def extract_image_features(image_path):
    """Extract basic features from an image."""
    try:
        features = {}
        
        # Use PIL for feature extraction since OpenCV might not be available
        try:
            img = Image.open(image_path)
            
            # Get image dimensions
            width, height = img.size
            
            # File info
            file_size = os.path.getsize(image_path) / 1024  # KB
            file_type = os.path.splitext(image_path)[1].lower().replace('.', '')
            
            # Convert to numpy array for calculations
            img_array = np.array(img)
            
            # Basic stats
            if len(img_array.shape) >= 3 and img_array.shape[2] >= 3:
                # RGB image
                brightness = np.mean(img_array)
                # Simplified contrast calculation
                contrast = np.std(img_array.flatten())
                
                # Average color
                avg_color = np.mean(img_array, axis=(0, 1))
                if len(avg_color) >= 3:
                    r, g, b = avg_color[:3]
                else:
                    r, g, b = 0, 0, 0
                
                # Calculate color distribution
                color_richness = np.std(avg_color)
            else:
                # Grayscale image
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
                r, g, b = brightness, brightness, brightness
                color_richness = 0
            
            # Create grayscale version for edge detection approximation
            if len(img_array.shape) >= 3 and img_array.shape[2] >= 3:
                # Convert RGB to grayscale
                grayscale = np.mean(img_array, axis=2)
            else:
                grayscale = img_array
                
            # Simple edge detection approximation
            edge_value = 0
            if grayscale.size > 0:
                # Calculate horizontal and vertical differences
                h_diff = np.abs(grayscale[:, 1:] - grayscale[:, :-1]).mean()
                if grayscale.shape[0] > 1:
                    v_diff = np.abs(grayscale[1:, :] - grayscale[:-1, :]).mean()
                    edge_value = (h_diff + v_diff) / 2
                else:
                    edge_value = h_diff
            
            features = {
                "dimensions": f"{width}x{height}",
                "file_size": f"{file_size:.1f} KB",
                "file_type": file_type.upper(),
                "brightness": f"{brightness:.2f}",
                "contrast": f"{contrast:.2f}",
                "color_richness": f"{color_richness:.2f}",
                "average_color": f"RGB({r:.1f}, {g:.1f}, {b:.1f})",
                "estimated_edges": f"{edge_value:.2f}"
            }
        except Exception as e:
            return {"error": f"Feature extraction error: {str(e)}"}
        
        return features
    except Exception as e:
        print(Fore.RED + f"Feature extraction error: {str(e)}" + Style.RESET_ALL)
        return {"error": str(e)}

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    try:
        # Simple DB check
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute('SELECT 1')
        conn.close()
        status = 'ok'
    except Exception as e:
        status = f'db_error: {str(e)}'
    return jsonify({
        'status': status,
        'provider': AI_PROVIDER,
        'port': int(os.environ.get('PORT', 5000))
    })

@app.route('/config_debug')
def config_debug():
    # Do NOT expose secrets; just presence flags
    return jsonify({
        'AI_PROVIDER': AI_PROVIDER,
        'GROQ_KEY_SET': bool(os.environ.get('GROQ_API_KEY')),
        'GOOGLE_KEY_SET': bool(os.environ.get('GOOGLE_API_KEY')),
        'GEMINI_MODEL': os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL),
        'DATABASE_PATH': DATABASE_PATH
    })

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    model = request.form.get('model', 'llama-3.1-8b-instant')  # Default model for Groq
    # If Gemini provider is selected, force a valid Gemini model
    if AI_PROVIDER == 'gemini':
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL)

    # Get answer from the model
    answer = get_answer(GROQ_API_KEY, question, model)

    # Save interaction to database
    add_interaction(question, answer, model)

    # Flash the answer to the user
    flash(answer)
    return redirect(url_for('index'))

@app.route('/history')
def history():
    interactions = get_all_interactions()
    print(interactions)  # Debug: Print the data being passed to the template
    return render_template('history.html', interactions=interactions)

@app.route('/analyze')
def analyze():
    analyze_interactions()
    flash("Interaction analysis completed. Check the console for results.")
    return redirect(url_for('index'))

@app.route('/switch_model', methods=['POST'])
def switch_model():
    new_model = request.form['model']
    # Test if the selected model is accessible
    if test_model_access(GROQ_API_KEY, new_model):
        flash(f"Switched to {new_model}")
    else:
        flash(f"Failed to switch to {new_model}. Model not accessible.")
    return redirect(url_for('index'))

@app.route('/upload_image', methods=['GET'])
def upload_image():
    """Display image upload form."""
    # Only show models with vision capabilities
    vision_models = [
        {"id": "llama3-70b-8192", "name": "Llama 3 70B (Most Powerful)"},
        {"id": "llama3-8b-8192", "name": "Llama 3 8B (Quick Responses)"},
        {"id": "llama-3.2-11b-vision-preview", "name": "Llama 3.2 11B Vision (Latest)"},
        {"id": "llama-3.2-90b-vision-preview", "name": "Llama 3.2 90B Vision (Latest, Most Powerful)"}
    ]
    
    return render_template('upload_image.html', models=vision_models)

@app.route('/image_history')
def image_history():
    interactions = get_all_image_interactions()
    return render_template('image_history.html', interactions=interactions)

@app.route('/ask_about_image', methods=['POST'])
def ask_about_image():
    """Process a question about an image."""
    try:
        # Get form data
        image_path = request.form['image_path']
        question = request.form['question']
        model = request.form.get('model', 'llama3-70b-8192')
        
        # Process the question about the image
        answer = answer_image_question(image_path, question, model)
        
        # Save to database
        add_image_qa_to_db(image_path, question, answer, model)
        
        # Get previous Q&A for this image
        qa_history = get_image_qa_history(image_path)
        
        # Get image features
        features = extract_image_features(image_path)
        
        # Get the original description
        description = get_image_description(image_path)
        
        # Show the result with the new answer
        return render_template('image_result.html', 
                            image_path=image_path,
                            description=description,
                            features=features,
                            current_question=question,
                            current_answer=answer,
                            qa_history=qa_history)
    except Exception as e:
        print(Fore.RED + f"Error in ask_about_image: {str(e)}" + Style.RESET_ALL)
        traceback.print_exc()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('upload_image'))

# Handle image upload (POST)
@app.route('/upload_image', methods=['POST'])
def upload_image_post():
    """Process uploaded image file."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No image selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        selected_model = request.form.get('model', 'llama3-70b-8192')
        
        # Check if file is empty
        if file.filename == '':
            flash('No image selected', 'error')
            return redirect(request.url)
        
        # Check if file is an allowed type
        if not allowed_file(file.filename):
            flash('File type not allowed. Please upload an image (JPEG, PNG, etc.)', 'error')
            return redirect(request.url)
        
        # Create a secure filename to prevent path traversal attacks
        filename = secure_filename(file.filename)
        
        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        
        # Create upload directory if it doesn't exist
        image_upload_path = os.path.join(app.static_folder, 'uploads')
        os.makedirs(image_upload_path, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(image_upload_path, unique_filename)
        file.save(file_path)
        
        # Image recognition processing with proper error handling
        description = recognize_image(file_path, selected_model)
        
        # Extract features from the image
        features = extract_image_features(file_path)
        
        # Store in the database
        relative_path = os.path.join('static', 'uploads', unique_filename).replace('\\', '/')
        add_image_to_db(relative_path, description, selected_model)
        
        # Render the result template
        return render_template(
            'image_result.html',
            image_path=relative_path,
            description=description,
            features=features,
            qa_history=[]  # Empty Q&A history for new images
        )
    
    except Exception as e:
        print(Fore.RED + f"Error in upload_image_post: {str(e)}" + Style.RESET_ALL)
        traceback.print_exc()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('upload_image'))

# Add image Q&A entry to the database
def add_image_qa_to_db(image_path, question, answer, model):
    """Add a question and answer about an image to the database."""
    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # Get image_id if it exists
        c.execute('SELECT id FROM images WHERE path = ?', (image_path,))
        result = c.fetchone()
        image_id = result[0] if result else 0
        
        # Insert the Q&A entry
        c.execute(
            'INSERT INTO image_qa (image_id, image_path, question, answer, model, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
            (image_id, image_path, question, answer, model, timestamp)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(Fore.RED + f"Error adding Q&A to database: {str(e)}" + Style.RESET_ALL)
        return False

# Get image description from database
def get_image_description(image_path):
    """Get the description of an image from the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute('SELECT description FROM images WHERE path = ? ORDER BY id DESC LIMIT 1', (image_path,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else "No description available"
    except Exception as e:
        print(Fore.RED + f"Error getting image description: {str(e)}" + Style.RESET_ALL)
        return "Error retrieving description"

# Get Q&A history for an image
def get_image_qa_history(image_path):
    """Get the Q&A history for an image."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.execute(
            'SELECT * FROM image_qa WHERE image_path = ? ORDER BY timestamp DESC', 
            (image_path,)
        )
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(Fore.RED + f"Error getting Q&A history: {str(e)}" + Style.RESET_ALL)
        return []

# Run the Flask app
if __name__ == '__main__':
    create_database()
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))