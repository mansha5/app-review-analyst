import json
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
ANALYSIS_DIR = "analysis"

SYSTEM_PROMPT = """You are a senior product analyst.
Return ONLY a single valid JSON object. No markdown, no arrays, no explanation."""

def parse_llm_response(raw):
    raw = raw.strip()
    # strip markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    # if model returned an array, take first element
    if raw.startswith("["):
        raw = json.loads(raw)
        return raw[0] if isinstance(raw, list) else raw
    return json.loads(raw)

def extract_topic_insight(app_name, topic):
    reviews_text = "\n".join(f"- {r}" for r in topic["sample_reviews"])

    prompt = f"""App: {app_name}
Keywords: {', '.join(topic['keywords'])}
Avg rating: {topic['avg_rating']} / 5
Review count: {topic['review_count']}
Reviews:
{reviews_text}

Reply with ONLY this JSON object (no other text):
{{"label":"5 word max label","category":"bug|complaint|feature_request|praise|other","root_cause":"one sentence","priority":"high|medium|low","priority_reason":"one sentence","pm_action":"one concrete action"}}

Priority: high if rating<2.5 and count>30, low if positive, else medium."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500,  # increased from 300
    )

    raw = response.choices[0].message.content.strip()
    return parse_llm_response(raw)

def process_app(app_name):
    path = f"{ANALYSIS_DIR}/{app_name}_topics.json"
    if not os.path.exists(path):
        return

    with open(path) as f:
        topics = json.load(f)

    print(f"\n{app_name} ({len(topics)} topics)...")

    enriched = []
    failed = 0
    for topic in topics:
        try:
            insight = extract_topic_insight(app_name, topic)
            enriched.append({**topic, **insight})
            print(f"  topic {topic['topic_id']:>2} → "
                  f"[{insight['category']}] {insight['label']} | "
                  f"priority: {insight['priority']}")
            time.sleep(0.5)
        except Exception as e:
            failed += 1
            print(f"  topic {topic['topic_id']:>2} → failed: {e}")
            enriched.append(topic)

    out_path = f"{ANALYSIS_DIR}/{app_name}_insights.json"
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"  done: {len(enriched)-failed} succeeded, {failed} failed")

if __name__ == "__main__":
    apps = sorted([
        f.replace("_topics.json", "")
        for f in os.listdir(ANALYSIS_DIR)
        if f.endswith("_topics.json")
    ])

    for app in apps:
        process_app(app)

    print("\nAll done.")