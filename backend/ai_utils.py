import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


FEW_SHOT_EXAMPLES = """Here are some examples of how to analyze reviews:

Review: "Mahao adum haowe yam items loina, adubu order lakpada matam change khara. Everything else is alright"
{"sentiment": "neutral", "aspects": ["food_quality", "service"]}

Review: "Special Fried chicken try toujabane, yaamna tasty oiye. reccomended to everyone"
{"sentiment": "positive", "aspects": ["food_quality"]}

Review: "Price henna high oiye, taste su price ka chanade. ambience is good tho"
{"sentiment": "negative", "aspects": ["price", "food_quality"]}
"""

def analyze_review(review_text: str):
    prompt = f"""{FEW_SHOT_EXAMPLES}
Now analyze this business review, which may mix Manipuri and English.
Note: a review can be genuinely mixed — praising one thing while criticizing another. Weigh the overall balance rather than defaulting to positive just because it opens well.

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