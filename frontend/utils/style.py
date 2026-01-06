"""UI styling helpers for the Streamlit frontend."""

import streamlit as st


def apply_minimal_style() -> None:
    """Inject a restrained, minimalist style (no emojis, modern buttons)."""

    st.markdown(
        """
<style>
/* Palette (keep restrained) */
:root {
  --mq-bg: #f6f7f9;          /* light grey */
  --mq-surface: #ffffff;     /* cards/inputs */
  --mq-sidebar: #f2f4f7;     /* slightly tinted */
  --mq-border: #d0d5dd;
  --mq-primary: #667085;     /* neutral grey */
  --mq-primary-hover: #475467;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* App background */
html, body, .stApp {
  background-color: var(--mq-bg);
}

/* Sidebar background */
section[data-testid="stSidebar"] {
  background-color: var(--mq-sidebar);
}

/* Layout tightening */
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 2rem;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container {
  padding-top: 1rem;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
  border-radius: 10px;
  padding: 0.55rem 0.9rem;
  font-weight: 600;
  border-width: 1px;
  transition: transform 80ms ease-in-out, filter 120ms ease-in-out;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
  filter: brightness(0.98);
}
.stButton > button:active,
.stFormSubmitButton > button:active {
  transform: translateY(1px);
}

/* Neutralize Streamlit primary button (avoid red accent) */
button[kind="primary"],
button[data-testid="baseButton-primary"] {
  background-color: var(--mq-primary) !important;
  border-color: var(--mq-primary) !important;
  color: #ffffff !important;
}
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
  background-color: var(--mq-primary-hover) !important;
  border-color: var(--mq-primary-hover) !important;
}

/* Make secondary buttons look clean */
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
  background-color: var(--mq-surface) !important;
  border-color: var(--mq-border) !important;
  color: inherit !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
  border-radius: 10px;
}

/* Expanders */
details {
  border-radius: 12px;
  overflow: hidden;
}

/* Metrics */
div[data-testid="stMetric"] {
  border-radius: 12px;
}
</style>
        """,
        unsafe_allow_html=True,
    )
