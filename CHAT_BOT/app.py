"""
app.py — a Streamlit chat frontend for the IRS Tax Q&A bot.

It wraps the TaxChatbot from chatbot.py in a web chat interface. Streamlit
re-runs this whole file top-to-bottom on every interaction, so the bot and
the visible messages are kept in st.session_state to survive those reruns.

Run it with:
    uv run streamlit run app.py
    (or:  streamlit run app.py)
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page setup + a light visual identity (ocean palette, serif display header)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="IRS Tax Q&A", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
      /* Serif display header, ocean-blue accents to match the project deck */
      .app-title { font-family: Cambria, Georgia, serif; font-size: 2.3rem;
                   font-weight: 700; color: #065A82; margin-bottom: 0.1rem; }
      .app-sub   { color: #5A7184; font-size: 1rem; margin-bottom: 1.2rem; }
      .stChatMessage { border-radius: 12px; }
      /* Trim Streamlit's default top padding */
      .block-container { padding-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">IRS Tax Q&amp;A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Ask about tax brackets, deductions, or anything in the IRS documents. '
    "Follow-up questions work — the bot remembers the conversation.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Guard: the chat layer needs an OpenAI key. Fail with direction, not a stack trace.
# ---------------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "No OpenAI API key found. Add a line `OPENAI_API_KEY=sk-...` to a `.env` "
        "file in this folder, then restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: settings + reset. Import the bot lazily so a missing key doesn't
# trigger a heavy model load before the guard above can run.
# ---------------------------------------------------------------------------
from IRS import TaxChatbot  # noqa: E402  (import after the API-key guard)

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Passages to retrieve (top_k)", min_value=1, max_value=10, value=5,
                      help="How many context chunks to pull in per question.")
    st.caption("Model: gpt-4o-mini")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("bot", None)
        st.session_state.pop("messages", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Persistent state: one bot instance + the visible transcript, kept across reruns.
# ---------------------------------------------------------------------------
if "bot" not in st.session_state:
    st.session_state.bot = TaxChatbot()
    st.session_state.messages = []  # list of {"role", "content"} for display

# Keep the retrieval depth in sync with the slider (safe to change live).
st.session_state.bot.top_k = top_k

# ---------------------------------------------------------------------------
# Render the conversation so far.
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Handle a new question.
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a tax question..."):
    # Show + store the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show the reply.
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                answer = st.session_state.bot.chat(prompt)
            except Exception as error:  # network/API issues shouldn't crash the UI
                answer = f"Something went wrong while answering: {error}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})