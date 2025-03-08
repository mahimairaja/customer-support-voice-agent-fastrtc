import os
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from fastapi import FastAPI, WebSocket, Request
from ollama import Client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'ollama')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'llama3.2')

SYSTEM_MESSAGE = """Your name is Lisa. You are a voice assistant for London Electronics, located at Main Street, England. 
The hours are 8 AM to 8 PM daily, but we're closed on Sundays.

You're responsible for assisting customers with service-related inquiries with Televisions and Electronics.


- Maintain a fun, lighthearted vibe—say things like “Umm...”, “Well...”, or “I mean...” 
- Keep responses concise, as it's a voice conversation—avoid long monologues.
- Provide basic info about the electronics if asked, but steer the conversation efficiently toward service scheduling if needed."""

client = Client(
    host=OLLAMA_URL,
    headers={'x-some-header': 'some-value'}
)
groq_client = Groq()
app = FastAPI()

stt_model = get_stt_model()
tts_model = get_tts_model()

def echo(audio):
    prompt = stt_model.stt(audio)

    response_text = get_response_text(prompt, MODEL_NAME)

    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk

stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = await stream.handle_incoming_call(request)
    return response


@app.websocket("/telephone/handler")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and LLM."""
    await stream.telephone_handler(websocket)


def get_response_text(prompt, model_name):

    print(f"User: {repr(prompt)}")

    if model_name == "ollama":
        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response['message']['content']
    elif model_name == "groq":
        response_text = (
        groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,
            messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ]
        )
        .choices[0]
        .message.content
    )
        
    print(f"\nResponse: {response_text}\n")
    return response_text