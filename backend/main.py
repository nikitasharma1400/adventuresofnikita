import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "WanderInk AI Backend Running"}


@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    style: str = Form("Cinematic")
):

    image_bytes = await image.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are WanderInk AI, a cinematic travel storytelling assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Analyze this image and generate:

1. Mood of the photo
2. Aesthetic/vibe
3. 3 Instagram captions
4. 3 photo story ideas

Style: {style}

STRICT RULES:
- Number the captions as 1, 2, and 3.
- Start every caption on a completely new line.
- Leave one blank line between every caption.
- Return ONLY the captions in numbered format.
- Do NOT write headings for captions.
- Keep captions cinematic, modern, poetic, aesthetic, and non-cringe.
- Keep them short to medium length.

EXACT CAPTION FORMAT:

1. first caption here

2. second caption here

3. third caption here
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.content_type};base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.8,
    )

    result = response.choices[0].message.content.strip()

    return {
        "result": result
    }