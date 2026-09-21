import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_review(review_text: str):
    prompt = f"""Analyze this business review, which may mix Manipuri and English.

Review: "{review_text}"

Reply with ONLY valid JSON in this exact format, no other text:
{{
  "sentiment": "positive" or "negative" or "neutral",
  "aspects": ["service", "food_quality", "price", "ambience"] (include only aspects actually mentioned, can be empty list)
}}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )

    raw_output = response.choices[0].message.content
    return json.loads(raw_output)


result = analyze_review(
    "Staff yamna nungaiye, cafe su yam cozy oiye"
)
print(result)
print(type(result))