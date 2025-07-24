import os
import requests
import streamlit as st

# Get Groq API key from environment variable
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set the GROQ_API_KEY environment variable")
    st.stop()

groq_url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}

def chat_with_groq(message):
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7
    }
    response = requests.post(groq_url, json=payload, headers=headers)
    if response.status_code != 200:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

st.title("Groq Chatbot")

user_input = st.text_input("Enter your message:", "")

if st.button("Send") and user_input.strip():
    with st.spinner("Waiting for Groq AI response..."):
        try:
            answer = chat_with_groq(user_input)
            if answer:
                st.markdown(f"**Groq AI:** {answer}")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
