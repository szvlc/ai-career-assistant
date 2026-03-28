from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# Load environment variables
load_dotenv(override=True)

# Send notification (lead or unknown question)
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

# Tool: save user contact details
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording lead: {name}, email: {email}, notes: {notes}")
    return {"recorded": "ok"}

# Tool: save unanswered questions
def record_unknown_question(question):
    push(f"Unknown question: {question}")
    return {"recorded": "ok"}

# Tool definitions
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user wants to stay in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The user's email address"
            },
            "name": {
                "type": "string",
                "description": "The user's name if provided"
            },
            "notes": {
                "type": "string",
                "description": "Additional context about the conversation"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool if the assistant cannot answer a question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that could not be answered"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

class Me:

    def __init__(self):
        # Gemini endpoint (OpenAI-compatible)
        self.GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.openai = OpenAI(
            base_url=self.GEMINI_BASE_URL,
            api_key=self.GOOGLE_API_KEY
        )

        self.name = "Aleksander Szulc"

        # File paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "me", "linkedin.pdf")
        summary_path = os.path.join(current_dir, "me", "summary.txt")

        # Load LinkedIn profile
        self.linkedin = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except:
            self.linkedin = "LinkedIn data not available."

        # Load summary
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                self.summary = f.read()
        except:
            self.summary = "Summary not available."

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"Tool called: {tool_name}", flush=True)

            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })

        return results

    def system_prompt(self):
        system_prompt = f"""
You are acting as {self.name}.
IMPORTANT:
Always respond in English.
You are answering questions on {self.name}'s personal website.
Your role is to represent {self.name} professionally to potential recruiters,
clients, or collaborators.
Focus on:
- career
- skills
- experience
- projects
- background
If you don't know the answer, use the record_unknown_question tool.
If the user is interested or engaged in conversation,
ask if they would like to stay in touch and collect their email
using the record_user_details tool.
"""

        system_prompt += f"\n\n## Summary:\n{self.summary}\n"
        system_prompt += f"\n## LinkedIn Profile:\n{self.linkedin}\n"

        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [
            {"role": "user", "content": message}
        ]

        done = False

        while not done:
            response = self.openai.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",
                messages=messages,
                tools=tools
            )

            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls

                results = self.handle_tool_call(tool_calls)

                messages.append(message)
                messages.extend(results)
            else:
                done = True

        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()

    gr.ChatInterface(
        me.chat,
        type="messages",
        title="AI Career Assistant",
        description="Chat with my AI assistant to learn more about my experience, skills, and projects.",
        theme="soft"
    ).launch()
