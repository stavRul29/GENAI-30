import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_daily_haiku(mood: str, highlight: str) -> str:

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction="""You are a mindful haiku poet.
When given someone's mood and a highlight from their day,
you write exactly one haiku (5-7-5 syllables).
Respond with ONLY the haiku — no title, no explanation."""
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(
                f"My mood today: {mood}\nHighlight of my day: {highlight}"
            )
            return response.text

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait = 60
                print(f"Rate limit hit — waiting {wait} seconds before retry {attempt+1}/{max_attempts}...")
                time.sleep(wait)
            else:
                raise e

    return "Could not generate haiku — please try again in a few minutes."

def main():
    print("\n--- Day 1: Hello, Gemini API (Free) ---\n")

    mood = input("How are you feeling today? ")
    highlight = input("What was one highlight of your day? ")

    print("\nGenerating your haiku...\n")
    haiku = get_daily_haiku(mood, highlight)

    print("=" * 30)
    print(haiku)
    print("=" * 30)

    print("\n--- What just happened ---")
    print("Model used  : gemini-2.0-flash")
    print("System msg  : defined the AI persona + rules")
    print("User msg    : your mood + highlight")
    print("Cost        : Rs. 0.00 — free tier")

if __name__ == "__main__":
    main()
 