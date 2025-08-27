import sys
import os
import requests
from colorama import init, Fore, Style
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

init(autoreset=True)
load_dotenv()

def call_gemini(api_key, model, text):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": api_key}
        payload = {
            "contents": [
                {"parts": [{"text": text}]}
            ]
        }
        resp = requests.post(url, params=params, json=payload, timeout=60)
        if resp.status_code != 200:
            return None, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None, "Model returned no candidates"
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip(), None
        return None, "Unexpected response format"
    except Exception as e:
        return None, str(e)

def model_selection_menu():
    print(Fore.MAGENTA + "\nSelect a Gemini model for this question:" + Style.RESET_ALL)
    print("1. Gemini 2.5 Pro - Advanced reasoning, best for coding and research")
    print("2. Gemini 2.5 Flash - Fast, cost-efficient multimodal tasks")
    print("3. Gemini 2.5 Flash-Lite - Lightweight, cheaper, high-throughput")
    print("4. Gemini Nano - On-device (not available via API key)")
    choice = input(Fore.CYAN + "Enter 1-4 or model id: " + Style.RESET_ALL).strip()
    mapping = {
        "1": "gemini-2.5-pro",
        "2": "gemini-2.5-flash",
        "3": "gemini-2.5-flash-lite",
        "4": "gemini-nano"
    }
    return mapping.get(choice, choice)

def main():
    print(Fore.GREEN + "Gemini CLI (API-key only, no browser/chat fallback)" + Style.RESET_ALL)
    print(Fore.YELLOW + "Type 'exit' to quit, 'help' for options." + Style.RESET_ALL)

    history = []
    while True:
        try:
            question = input(Fore.CYAN + "Enter your question: " + Style.RESET_ALL).strip()
            if not question:
                continue
            if question.lower() == 'exit':
                print(Fore.YELLOW + "Goodbye." + Style.RESET_ALL)
                break
            if question.lower() == 'help':
                print("Commands: exit, help, history, clear")
                continue
            if question.lower() == 'history':
                if not history:
                    print("No history.")
                else:
                    for i, h in enumerate(history, 1):
                        print(f"{i}. [{h['model']}] {h['question']} -> {h['answer'][:80]}")
                continue
            if question.lower() == 'clear':
                history.clear()
                print("History cleared.")
                continue

            model = model_selection_menu()
            if model == "gemini-nano":
                print(Fore.RED + "Error: Gemini Nano is on-device and not accessible via API key." + Style.RESET_ALL)
                continue

            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if not api_key:
                api_key = input(Fore.CYAN + "Enter your GOOGLE_API_KEY: " + Style.RESET_ALL).strip()

            answer, err = call_gemini(api_key, model, question)
            if err:
                print(Fore.RED + f"Error: {err}" + Style.RESET_ALL)
                continue

            print(Fore.GREEN + "\nAnswer:" + Style.RESET_ALL)
            print(answer)
            print(Fore.BLUE + "-" * 40 + Style.RESET_ALL)

            history.append({"question": question, "answer": answer, "model": model})
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nInterrupted. Type 'exit' to quit." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Unexpected error: {str(e)}" + Style.RESET_ALL)

if __name__ == '__main__':
    main()