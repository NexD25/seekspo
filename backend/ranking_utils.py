from sqlalchemy.orm import Session
from models import Business, Review
from ai_utils import client


SENTIMENT_SCORES = {"positive": 1, "neutral": 0, "negative": -1}


def get_business_score(business_id: int, db: Session):
    reviews = db.query(Review).filter(Review.business_id == business_id).all()
    scored_reviews = [r for r in reviews if r.sentiment in SENTIMENT_SCORES]

    if not scored_reviews:
        return None

    total = sum(SENTIMENT_SCORES[r.sentiment] for r in scored_reviews)
    return total / len(scored_reviews)

def get_business_ranking(business_id: int, db: Session):
    business = db.query(Business).filter(Business.id == business_id).first()
    if not business:
        return None

    my_score = get_business_score(business_id, db)
    if my_score is None:
        return {"score": None, "percentile": None, "compared_to": 0, "reasoning": None}

    competitors = db.query(Business).filter(
        Business.category == business.category,
        Business.area == business.area,
        Business.id != business_id,
    ).all()

    competitor_scores = []
    for comp in competitors:
        score = get_business_score(comp.id, db)
        if score is not None:
            competitor_scores.append(score)

    if not competitor_scores:
        return {"score": my_score, "percentile": None, "compared_to": 0, "reasoning": None}

    better_than = sum(1 for s in competitor_scores if my_score > s)
    percentile = round((better_than / len(competitor_scores)) * 100)

    reviews = db.query(Review).filter(Review.business_id == business_id).all()
    all_aspects = [a for r in reviews if r.aspects for a in r.aspects.split(",")]
    reviews_summary = ", ".join(set(all_aspects)) if all_aspects else "no specific aspects mentioned"

    reasoning = generate_ranking_reasoning(business.name, percentile, reviews_summary)

    return {"score": my_score, "percentile": percentile, "compared_to": len(competitor_scores), "reasoning": reasoning}

def generate_ranking_reasoning(business_name: str, percentile: int, reviews_summary: str):
    prompt = f"""Business: {business_name}
This business ranks in the {percentile}th percentile among similar businesses in its area, based on customer review sentiment.
Here is a summary of what reviews mention: {reviews_summary}

Write ONE short, natural sentence (like a recommendation blurb) explaining why this business ranks where it does. Do not mention percentiles or numbers directly - make it sound natural, like a human summary.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()