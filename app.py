from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# 1. Wczytywanie zmiennych środowiskowych
load_dotenv(override=True)

# --- NARZĘDZIA (TOOLS) ---

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

# --- DEFINICJE JSON DLA NARZĘDZI ---

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Additional context about the conversation"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

# --- KLASA GŁÓWNA ---

class Me:
    def __init__(self):
        # Inicjalizacja klienta OpenAI (używa klucza z .env automatycznie)
        self.openai = OpenAI() 
        self.name = "Aleksander Szulc" 
        
        # Ustalanie ścieżek do Twoich plików
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "me", "linkedin.pdf")
        summary_path = os.path.join(current_dir, "me", "summary.txt")

        # Wczytywanie LinkedIn PDF
        self.linkedin = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except Exception as e:
            print(f"Błąd wczytywania PDF: {e}")
            self.linkedin = "Brak danych z LinkedIn."

        # Wczytywanie Summary TXT
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                self.summary = f.read()
        except Exception as e:
            print(f"Błąd wczytywania Summary: {e}")
            self.summary = "Brak podsumowania."

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        prompt = f"""You are acting as {self.name}. You are answering questions on your personal website.
Your responsibility is to represent yourself faithfully based on the following background.
Be professional, engaging, and polite.

## Summary:
{self.summary}

## LinkedIn Profile:
{self.linkedin}

If you don't know an answer, use the 'record_unknown_question' tool. 
If the user wants to stay in touch, use 'record_user_details' to get their email."""
        return prompt
    
    def chat(self, message, history):
        try:
            messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
            done = False
            while not done:
                # Korzystamy z modelu OpenAI
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=messages, 
                    tools=tools
                )
                
                if response.choices[0].finish_reason == "tool_calls":
                    msg_obj = response.choices[0].message
                    tool_calls = msg_obj.tool_calls
                    results = self.handle_tool_call(tool_calls)
                    messages.append(msg_obj)
                    messages.extend(results)
                else:
                    done = True
                    
            return response.choices[0].message.content
        except Exception as e:
            print(f"BŁĄD OPENAI: {e}")
            return f"Błąd połączenia z OpenAI: {str(e)}"

# --- URUCHOMIENIE ---

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()