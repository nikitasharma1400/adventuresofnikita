import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ask_groq(prompt: str, system: str = "You are WanderInk AI."):
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


@app.get("/")
def home():
    return {"message": "WanderInk AI Backend Running"}


@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    style: str = Form("Cinematic"),
):
    image_bytes = await image.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are WanderInk AI, a cinematic travel storytelling assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Analyze this image and generate:

Mood:
Aesthetic/Vibe:
Visual Details:
Best Caption Style:
3 Instagram Captions:
3 Story Ideas:

Selected Style: {style}

Rules:
- Keep it cinematic, modern, poetic, useful, and non-cringe.
- Captions must be numbered.
- Story ideas must be practical for Instagram carousel/reel.
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.content_type};base64,{encoded_image}"
                        },
                    },
                ],
            },
        ],
        temperature=0.8,
    )

    return {"result": response.choices[0].message.content.strip()}


@app.post("/recommend-style")
async def recommend_style(image: UploadFile = File(...)):
    image_bytes = await image.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a visual aesthetic expert for photographers and creators.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Analyze this image and recommend the best caption/storytelling style.

Return:
1. Best Style
2. Why it fits
3. 3 caption direction ideas
4. Suggested edit vibe
5. Suggested music/reel mood

Keep it concise and creator-friendly.
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            },
        ],
        temperature=0.7,
    )

    return {"recommendation": response.choices[0].message.content.strip()}


@app.post("/generate-blog")
def generate_blog(data: dict):
    memories = data.get("memories", [])

    prompt = f"""
Turn these travel/photo memories into a beautiful short blog/story post.

Memories:
{memories}

Return:
1. Title
2. Short poetic intro
3. Story body
4. Instagram caption
5. 5 hashtags

Style: cinematic, personal, warm, not cringe, not AI-like.
"""

    return {
        "blog": ask_groq(
            prompt,
            "You are a cinematic travel writer and visual storytelling assistant.",
        )
    }


@app.post("/carousel-sequence")
def carousel_sequence(data: dict):
    memories = data.get("memories", [])

    prompt = f"""
Arrange these photo memories into the best Instagram carousel sequence.

Memories:
{memories}

Return:
1. Opening slide
2. Middle flow
3. Closing slide
4. Carousel caption
5. Why this order works

Keep it practical and creator-friendly.
"""

    return {
        "sequence": ask_groq(
            prompt,
            "You are a visual storytelling strategist.",
        )
    }


@app.post("/recommendations")
def recommendations(data: dict):
    memories = data.get("memories", [])

    prompt = f"""
Based on these saved photo memories, create personalized creator recommendations.

Memories:
{memories}

Return:
1. Your strongest visual style
2. Best caption style to use more
3. 5 future photo ideas
4. 3 carousel ideas
5. 3 reel ideas
6. Posting strategy

Keep it actionable and specific.
"""

    return {
        "recommendations": ask_groq(
            prompt,
            "You are an Instagram growth strategist for photographers.",
        )
    }