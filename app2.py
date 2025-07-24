import os
import requests
import streamlit as st

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set the GROQ_API_KEY environment variable")
    st.stop()

groq_url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}

def chat_with_groq(messages):
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.7,
    }
    response = requests.post(groq_url, json=payload, headers=headers)
    if response.status_code != 200:
        st.error(f"Error: {response.status_code} - {response.text}")
        return ""
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# State: chat history and answer-wait flag
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "latest_question" not in st.session_state:
    st.session_state.latest_question = ""

st.title("Groq AI Chat")

# Process answer on rerun if needed
if st.session_state.awaiting_response and st.session_state.latest_question:
    # Add the user's new message to chat history
    st.session_state.chat_history.append({"role": "user", "content": st.session_state.latest_question})
    with st.spinner("Groq AI is typing..."):
        answer = chat_with_groq(st.session_state.chat_history)
    if answer:
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    # Reset flags/input for next turn
    st.session_state.awaiting_response = False
    st.session_state.latest_question = ""

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Groq AI:** {msg['content']}")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submit = st.form_submit_button("Send")

    if submit and user_input.strip():
        # Store the question and set the flag for answer processing on rerun
        st.session_state.latest_question = user_input
        st.session_state.awaiting_response = True
        st.rerun()
