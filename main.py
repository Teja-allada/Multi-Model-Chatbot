import sys
import requests
from colorama import init, Fore, Style
import sqlite3
import pandas as pd
import time
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set UTF-8 encoding for stdout
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # Fallback for environments where reconfigure isn't available
    pass

# Initialize Colorama
init(autoreset=True)

# API constants
GROQ_API_KEY = "you_grok_api_key"  # Your API key
MAX_RETRIES = 3
LONG_TIMEOUT = 120  # 2 minutes for code generation

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

    conn.commit()
    conn.close()

def get_all_interactions():
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interactions")
    interactions = c.fetchall()
    conn.close()
    return interactions

def add_interaction(question, answer, model):
    conn = sqlite3.connect('interactions.db')
    c = conn.cursor()
    c.execute("INSERT INTO interactions (question, answer, model) VALUES (?, ?, ?)",
              (question, answer, model))
    conn.commit()
    conn.close()

def view_interactions():
    interactions = get_all_interactions()
    if len(interactions) == 0:
        print(Fore.RED + "No interactions found." + Style.RESET_ALL)
    else:
        print(Fore.MAGENTA + "\nQuestion-Answer Interactions:" + Style.RESET_ALL)
        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
        for interaction in interactions:
            print(f"ID: {interaction[0]}")
            print(f"Question: {interaction[1]}")
            print(f"Answer: {interaction[2][:100]}..." if len(interaction[2]) > 100 else f"Answer: {interaction[2]}")
            print(f"Model: {interaction[3] if len(interaction) > 3 else 'unknown'}")
            print(f"Timestamp: {interaction[4] if len(interaction) > 4 else 'unknown'}")
            print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

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

# Display Intro
def display_intro():
    print(Fore.CYAN + """

 __  __        _____  _____ _______
|  \/  |      / ___|  __ \_   __|
| \  / |_   | |  __| |_) | | |
| |\/| | | | | | |_ |  ___/  | |
| |  | | || | |_| | |      | |
||  ||\, |\||      |_|
         __/ |
        |_/

MyGPT V4.0 Ultimate Code Edition
Enhanced for complete code generation with DeepSeek and other models!
    """ + Style.RESET_ALL)

# History Management
def find_last_question(history):
    for i in range(len(history) - 1, -1, -1):
        if 'question' in history[i]:
            return history[i]['question']
    return None

def develop_question(history):
    last_question = find_last_question(history)
    if last_question:
        new_question = input(Fore.YELLOW + f"\nLast Question: {last_question}\nDevelop your question: " + Style.RESET_ALL)
        return new_question
    else:
        print(Fore.RED + "\nNo last question available." + Style.RESET_ALL)
        return None

def search_history(history, keyword):
    found_items = []
    for item in history:
        if keyword.lower() in item['question'].lower() or keyword.lower() in item['answer'].lower():
            found_items.append(item)
    return found_items

def save_history_to_file(history, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in history:
            file.write(f"Question: {item['question']}\n")
            file.write(f"Answer: {item['answer']}\n")
            file.write("-" * 40 + "\n")

def load_history_from_file(filename):
    history = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines)-1, 3):
                question = lines[i].replace("Question: ", "").strip()
                answer = lines[i+1].replace("Answer: ", "").strip()
                history.append({"question": question, "answer": answer})
        return history
    except FileNotFoundError:
        print(Fore.RED + f"File {filename} not found." + Style.RESET_ALL)
        return []
    except Exception as e:
        print(Fore.RED + f"Error loading history: {str(e)}" + Style.RESET_ALL)
        return []

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

# Display available models by provider
def list_available_models():
    print(Fore.MAGENTA + "\nAvailable Models by Provider:" + Style.RESET_ALL)
    print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)

    print(Fore.YELLOW + "Alibaba Cloud:" + Style.RESET_ALL)
    print("1. qwen-2.5-32b (Large general purpose model)")
    print("2. qwen-2.5-coder-32b (Specialized for coding tasks)")

    print(Fore.YELLOW + "\nDeepSeek:" + Style.RESET_ALL)
    print("3. deepseek-r1-distill-qwen-32b (DeepSeek distilled from Qwen)")
    print("4. deepseek-r1-distill-llama-70b (DeepSeek distilled from LLaMA)")

    print(Fore.YELLOW + "\nGoogle:" + Style.RESET_ALL)
    print("5. gemma2-9b-it (Google's instruction-tuned model)")

    print(Fore.YELLOW + "\nMeta (LLaMA):" + Style.RESET_ALL)
    print("6. llama-3.1-8b-instant (Fast responses)")
    print("7. llama-3.2-11b-vision-preview (Vision capabilities)")
    print("8. llama-3.2-1b-preview (Very small model)")
    print("9. llama-3.2-3b-preview (Small efficient model)")
    print("10. llama-3.2-90b-vision-preview (Largest vision model)")
    print("11. llama-3.3-70b-specdec (Specialized for detailed tasks)")
    print("12. llama-3.3-70b-versatile (Strong general purpose)")
    print("13. llama-guard-3-8b (Content moderation)")
    print("14. llama3-70b-8192 (Large with 8K context)")
    print("15. llama3-8b-8192 (Efficient with 8K context)")

    print(Fore.YELLOW + "\nHugging Face:" + Style.RESET_ALL)
    print("16. distil-whisper-large-v3-en (Audio transcription)")

    print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)

# Function to check which models are accessible with the API key
def test_api_models(api_key):
    print(Fore.YELLOW + "\nTesting API key with different models..." + Style.RESET_ALL)

    # A subset of models to test, prioritizing code-capable models
    test_models = [
        "deepseek-r1-distill-qwen-32b",  # Great for code
        "qwen-2.5-coder-32b",           # Specialized for code
        "llama-3.1-8b-instant",         # Fast, reliable
        "llama3-70b-8192"               # Large context
    ]

    accessible_models = []
    for model in test_models:
        print(f"Testing {model}...", end="")
        if test_model_access(api_key, model):
            print(Fore.GREEN + " Accessible" + Style.RESET_ALL)
            accessible_models.append(model)
        else:
            print(Fore.RED + " Not accessible" + Style.RESET_ALL)

    if accessible_models:
        print(Fore.GREEN + "\nYour API key can access these models:" + Style.RESET_ALL)
        for model in accessible_models:
            print(f"- {model}")

        # Return the best coding model available
        for preferred in ["deepseek-r1-distill-qwen-32b", "qwen-2.5-coder-32b", "llama3-70b-8192", "llama-3.1-8b-instant"]:
            if preferred in accessible_models:
                return preferred

        return accessible_models[0]  # Return first accessible model as default
    else:
        print(Fore.RED + "\nYour API key couldn't access any of the tested models." + Style.RESET_ALL)
        return "llama-3.1-8b-instant"  # Fallback to a commonly available model

# Display help menu
def display_help():
    print(Fore.MAGENTA + "\nMyGPT Commands:" + Style.RESET_ALL)
    print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)

    commands = [
        ("exit", "Exit the program"),
        ("help", "Show this help menu"),
        ("history", "Show conversation history"),
        ("clear", "Clear conversation history"),
        ("repeat", "Show the last question and answer"),
        ("search", "Search previous conversations"),
        ("save", "Save conversation history to a file"),
        ("load", "Load conversation history from a file"),
        ("view", "View database interactions"),
        ("analyze", "Analyze conversation patterns"),
        ("list_models", "List available models"),
        ("switch_model", "Switch to a different model"),
        ("current_model", "Show the currently active model"),
        ("test_models", "Test which models your API key can access"),
        ("code_mode", "Optimize next response for code generation"),
        ("develop", "Continue from your last question")
    ]

    for cmd, desc in commands:
        print(f"{Fore.CYAN}{cmd:<15}{Style.RESET_ALL}{desc}")

    print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)

# Main Function
def main():
    create_database()
    display_intro()

    # Set Groq API Key
    api_key = GROQ_API_KEY
    if not api_key:
        api_key = input(Fore.CYAN + "Enter your Groq API key: " + Style.RESET_ALL).strip()

    print(Fore.GREEN + "Groq API key loaded successfully." + Style.RESET_ALL)

    # Test which models the API key can access
    print(Fore.CYAN + "Checking which models your API key can access..." + Style.RESET_ALL)
    default_model = test_api_models(api_key)

    # Set the default model based on access test
    current_model = default_model
    print(Fore.GREEN + f"Using {current_model} as default model" + Style.RESET_ALL)
    print(Fore.YELLOW + "Type 'help' to see available commands" + Style.RESET_ALL)

    history = []
    last_question = None
    last_answer = None
    code_mode = False  # Toggle for code generation optimization

    while True:
        try:
            prompt = f"{'[CODE MODE] ' if code_mode else ''}Enter your question: "
            question = input(Fore.CYAN + prompt + Style.RESET_ALL).strip()

            if not question:
                continue

            # Command processing
            if question.lower() == 'exit':
                print(Fore.YELLOW + "MyGPT welcomes you - See you soon." + Style.RESET_ALL)
                break

            elif question.lower() == 'help':
                display_help()
                continue

            elif question.lower() == 'history':
                print(Fore.MAGENTA + "\nQuestion History:" + Style.RESET_ALL)
                print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
                if not history:
                    print("No history available.")
                else:
                    for idx, item in enumerate(history, start=1):
                        print(f"{idx}. Question: {item['question']}")
                        print(f"   Answer: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"   Answer: {item['answer']}")
                        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
                continue

            elif question.lower() == 'clear':
                history.clear()
                print(Fore.GREEN + "\nQuestion history cleared." + Style.RESET_ALL)
                continue

            elif question.lower() == 'repeat':
                if last_question and last_answer:
                    print(Fore.YELLOW + "\nLast Question:" + Style.RESET_ALL)
                    print(last_question)
                    print(Fore.YELLOW + "Last Answer:" + Style.RESET_ALL)
                    print(last_answer)
                    print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
                else:
                    print(Fore.RED + "\nNo last question and answer available." + Style.RESET_ALL)
                continue

            elif question.lower() == 'search':
                keyword = input(Fore.CYAN + "Enter keyword to search for: " + Style.RESET_ALL)
                if not keyword:
                    continue

                # Simple search in history
                results = []
                for item in history:
                    if keyword.lower() in item['question'].lower() or keyword.lower() in item['answer'].lower():
                        results.append(item)

                print(Fore.MAGENTA + f"\nSearch Results for '{keyword}':" + Style.RESET_ALL)
                print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

                if results:
                    for idx, item in enumerate(results, start=1):
                        print(f"{idx}. Question: {item['question']}")
                        print(f"   Answer: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"   Answer: {item['answer']}")
                        print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)
                else:
                    print(Fore.RED + "No matching questions or answers found." + Style.RESET_ALL)
                continue

            elif question.lower() == 'save':
                filename = input(Fore.CYAN + "Enter filename to save the question history (e.g., history.txt): " + Style.RESET_ALL)
                if not filename:
                    continue
                save_history_to_file(history, filename)
                print(Fore.GREEN + f"\nQuestion history saved to '{filename}'." + Style.RESET_ALL)
                continue

            elif question.lower() == 'load':
                filename = input(Fore.CYAN + "Enter filename to load the question history from: " + Style.RESET_ALL)
                if not filename:
                    continue
                loaded_history = load_history_from_file(filename)
                if loaded_history:
                    history = loaded_history
                    print(Fore.GREEN + f"\nQuestion history loaded from '{filename}'." + Style.RESET_ALL)
                continue

            elif question.lower() == 'view':
                view_interactions()
                continue

            elif question.lower() == 'analyze':
                analyze_interactions()
                continue

            elif question.lower() == 'list_models':
                list_available_models()
                continue

            elif question.lower() == 'test_models':
                test_api_models(api_key)
                continue

            elif question.lower() == 'code_mode':
                code_mode = not code_mode
                print(Fore.GREEN + f"Code mode is now {'ON' if code_mode else 'OFF'}" + Style.RESET_ALL)
                continue

            elif question.lower() == 'develop':
                new_question = develop_question(history)
                if new_question:
                    question = new_question
                else:
                    continue

            elif question.lower() == 'switch_model':
                list_available_models()
                model_choice = input(Fore.CYAN + "Enter model number (1-16) or model name: " + Style.RESET_ALL)

                # Dictionary mapping numbers to model names
                model_map = {
                    "1": "qwen-2.5-32b",
                    "2": "qwen-2.5-coder-32b",
                    "3": "deepseek-r1-distill-qwen-32b",
                    "4": "deepseek-r1-distill-llama-70b",
                    "5": "gemma2-9b-it",
                    "6": "llama-3.1-8b-instant",
                    "7": "llama-3.2-11b-vision-preview",
                    "8": "llama-3.2-1b-preview",
                    "9": "llama-3.2-3b-preview",
                    "10": "llama-3.2-90b-vision-preview",
                    "11": "llama-3.3-70b-specdec",
                    "12": "llama-3.3-70b-versatile",
                    "13": "llama-guard-3-8b",
                    "14": "llama3-70b-8192",
                    "15": "llama3-8b-8192",
                    "16": "distil-whisper-large-v3-en"
                }

                if model_choice in model_map:
                    new_model = model_map[model_choice]
                else:
                    # Assume the user entered a model name directly
                    new_model = model_choice

                # Test if the selected model is accessible
                print(f"Testing access to {new_model}...", end="")
                if test_model_access(api_key, new_model):
                    print(Fore.GREEN + " Success!" + Style.RESET_ALL)
                    current_model = new_model
                    print(Fore.GREEN + f"Switched to {current_model}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + " Failed." + Style.RESET_ALL)
                    print(Fore.RED + f"Your API key doesn't have access to the {new_model} model. Staying with {current_model}." + Style.RESET_ALL)
                continue

            elif question.lower() == 'current_model':
                print(Fore.GREEN + f"Currently using: {current_model}" + Style.RESET_ALL)
                continue

            # Process the user's question
            print(Fore.YELLOW + f"Processing your question with {current_model}..." + Style.RESET_ALL)

            # Check if it's a code-related question
            is_code = code_mode or is_code_question(question)
            if is_code:
                print(Fore.YELLOW + "Detected code-related question. Using optimized settings." + Style.RESET_ALL)

            # Get answer with potentially multiple attempts for complete code
            answer = get_answer(api_key, question, current_model, is_code)

            if answer:
                print(Fore.GREEN + "\nAnswer:" + Style.RESET_ALL)
                print(Fore.MAGENTA + answer + Style.RESET_ALL)
                print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

                # Save to database
                add_interaction(question, answer, current_model)

                # Add to history
                history.append({"question": question, "answer": answer})

                # Update last question and answer
                last_question = question
                last_answer = answer

                # Turn off code mode after use
                if code_mode:
                    code_mode = False
                    print(Fore.YELLOW + "Code mode automatically turned off." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Failed to get a response. Try using a different model." + Style.RESET_ALL)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nOperation interrupted. Type 'exit' to quit or continue with a new question." + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"An unexpected error occurred: {str(e)}" + Style.RESET_ALL)

if __name__ == '__main__':
    main()