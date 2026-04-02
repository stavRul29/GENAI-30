import streamlit as st
from groq import Groq

st.set_page_config(page_title="Daily Haiku Generator", page_icon="🍃")

st.title("🍃 Daily Haiku Generator")
st.write("Share your mood and a highlight from your day — get a haiku in return.")

mood = st.text_input("How are you feeling today?", placeholder="e.g. grateful, tired, excited")
highlight = st.text_input("What was one highlight of your day?", placeholder="e.g. had coffee with an old friend")

if st.button("Generate Haiku", disabled=not (mood and highlight)):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    with st.spinner("Writing your haiku..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a mindful haiku poet. "
                        "When given someone's mood and a highlight from their day, "
                        "you write exactly one haiku (5-7-5 syllables). "
                        "Respond with ONLY the haiku — no title, no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"My mood today: {mood}\nHighlight of my day: {highlight}"
                }
            ]
        )

    haiku = response.choices[0].message.content.strip()
    lines = [line.strip() for line in haiku.split("\n") if line.strip()]
    st.markdown("---")
    st.markdown("### ✨ Your Haiku")
    for line in lines:
        st.markdown(f"_{line}_")
    st.markdown("---")
    st.caption("Powered by Llama 3.3 via Groq · Runs in the cloud · Free")
