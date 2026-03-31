import ollama

def get_daily_haiku(mood: str, highlight: str) -> str:
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """You are a mindful haiku poet.
When given someone's mood and a highlight from their day,
you write exactly one haiku (5-7-5 syllables).
Respond with ONLY the haiku — no title, no explanation."""
            },
            {
                "role": "user",
                "content": f"My mood today: {mood}\nHighlight of my day: {highlight}"
            }
        ]
    )
    return response['message']['content']

def main():
    print("\n--- Day 1: Hello, Local AI (Ollama + Llama 3.2) ---\n")

    mood = input("How are you feeling today? ")
    highlight = input("What was one highlight of your day? ")

    print("\nGenerating your haiku...\n")
    haiku = get_daily_haiku(mood, highlight)

    print("=" * 30)
    print(haiku)
    print("=" * 30)

    print("\n--- What just happened ---")
    print("Model used  : llama3.2 (running locally on your machine)")
    print("System msg  : defined the AI persona + rules")
    print("User msg    : your mood + highlight")
    print("Cost        : Rs. 0.00 — ran entirely on your laptop")
    print("Data sent   : nothing left your machine")

if __name__ == "__main__":
    main()