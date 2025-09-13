import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification  # Added for the classifier
)
import spacy
import time
import random  # Added for fallback responses

# =============================
# MODEL AND CONFIGURATION SETUP
# =============================

# Hugging Face model IDs
GPT2_MODEL_ID = "Zlib2/ETCSCb_DistilGPT2"
CLASSIFIER_ID = "Zlib2/Query_Classifier_DistilBERT"  # ID for the new classifier model

# Random OOD Fallback Responses
fallback_responses = [
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets.",
    "Apologies, but this falls outside the scope of my support. I’m here if you need any help with event ticket issues.",
    "I'm sorry, but I cannot assist with this particular topic. If you have questions about event tickets, I’d be glad to help.",
    "I regret that I’m unable to provide assistance here. Please let me know how I can support you with event ticket matters.",
    "Unfortunately, I am not equipped to assist with this. If you need help with event tickets, I am here for that.",
    "I apologize, but I cannot help with this request. However, I’d be happy to assist with anything related to event tickets.",
    "I’m sorry, but I’m unable to support this request. If it’s about event tickets, I’ll gladly help however I can.",
    "This matter falls outside the assistance I can offer. Please let me know if you need help with event ticket-related inquiries.",
    "Regrettably, this is not something I can assist with. I’m happy to help with any event ticket questions you may have.",
    "I’m unable to provide support for this issue. However, I can assist with concerns regarding event tickets.",
    "I apologize, but I cannot help with this matter. If your inquiry is related to event tickets, I’d be more than happy to assist.",
    "I regret that I am unable to offer help in this case. I am, however, available for any event ticket-related questions.",
    "Unfortunately, I’m not able to assist with this. Please let me know if there’s anything I can do regarding event tickets.",
    "I'm sorry, but I cannot assist with this topic. However, I’m here to help with any event ticket concerns you may have.",
    "Apologies, but this request falls outside of my support scope. If you need help with event tickets, I’m happy to assist.",
    "I’m afraid I can’t help with this matter. If there’s anything related to event tickets you need, feel free to reach out.",
    "This is beyond what I can assist with at the moment. Let me know if there’s anything I can do to help with event tickets.",
    "Sorry, I’m unable to provide support on this issue. However, I’d be glad to assist with event ticket-related topics.",
    "Apologies, but I can’t assist with this. Please let me know if you have any event ticket inquiries I can help with.",
    "I’m unable to help with this matter. However, if you need assistance with event tickets, I’m here for you.",
    "Unfortunately, I can’t support this request. I’d be happy to assist with anything related to event tickets instead.",
    "I’m sorry, but I can’t help with this. If your concern is related to event tickets, I’ll do my best to assist.",
    "Apologies, but this issue is outside of my capabilities. However, I’m available to help with event ticket-related requests.",
    "I regret that I cannot assist with this particular matter. Please let me know how I can support you regarding event tickets.",
    "I’m sorry, but I’m not able to help in this instance. I am, however, ready to assist with any questions about event tickets.",
    "Unfortunately, I’m unable to help with this topic. Let me know if there's anything event ticket-related I can support you with."
]

# =============================
# MODEL LOADING FUNCTIONS
# =============================

@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    try:
        model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_ID, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load GPT-2 model from Hugging Face Hub. Error: {e}")
        return None, None

# --- NEW: Function to load the classifier model ---
@st.cache_resource(show_spinner=False)
def load_classifier_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier model from Hugging Face Hub. Error: {e}")
        return None, None

# --- NEW: Function to check if a query is Out-of-Domain (OOD) ---
def is_ood(query: str, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return pred_id == 1  # True if OOD (label 1)

# =============================
# ORIGINAL HELPER FUNCTIONS (UNCHANGED)
# =============================

# (The static_placeholders dictionary, replace_placeholders, extract_dynamic_placeholders,
#  and generate_response functions remain exactly as in your original script.)

static_placeholders = {
    # ... (unchanged dictionary content)
    "{{ASSISTANCE_SECTION}}" : "<b>Assistance Section</b>",
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# =============================
# CSS AND UI SETUP (UNCHANGED)
# =============================

st.markdown(
    """
<style>
.stButton>button { background: linear-gradient(90deg, #ff8a00, #e52e71); color: white !important; border: none; border-radius: 25px; padding: 10px 20px; font-size: 1.2em; font-weight: bold; cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease; display: inline-flex; align-items: center; justify-content: center; margin-top: 5px; width: auto; min-width: 100px; font-family: 'Times New Roman', Times, serif !important; }
.stButton>button:hover { transform: scale(1.05); box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); color: white !important; }
.stButton>button:active { transform: scale(0.98); }
* { font-family: 'Times New Roman', Times, serif !important; }
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) { background: linear-gradient(90deg, #29ABE2, #0077B6); color: white !important; }
.horizontal-line { border-top: 2px solid #e0e0e0; margin: 15px 0; }
div[data-testid="stChatInput"] { box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); border-radius: 5px; padding: 10px; margin: 10px 0; }
</style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='font-size: 43px;'>Advanced Event Ticketing Chatbot</h1>", unsafe_allow_html=True)

# =============================
# MODEL LOADING
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

example_queries = [
    "How do I buy a ticket?", "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?", "How can I find details about upcoming events?",
    "How do I contact customer service?", "How do I get a refund?", "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?", "How can I sell my ticket?"
]

if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources... Please wait..."):
        try:
            nlp = load_spacy_model()
            gpt2_model, gpt2_tokenizer = load_gpt2_model_and_tokenizer()
            clf_model, clf_tokenizer = load_classifier_model()

            if all([nlp, gpt2_model, gpt2_tokenizer, clf_model, clf_tokenizer]):
                st.session_state.models_loaded = True
                st.session_state.nlp = nlp
                st.session_state.model = gpt2_model
                st.session_state.tokenizer = gpt2_tokenizer
                st.session_state.clf_model = clf_model
                st.session_state.clf_tokenizer = clf_tokenizer
                st.rerun()
            else:
                st.error("Failed to load one or more models. Please refresh the page.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# =============================
# MAIN CHAT INTERFACE
# =============================

if st.session_state.models_loaded:
    st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")

    selected_query = st.selectbox(
        "Choose a query from examples:", ["Choose your question"] + example_queries,
        key="query_selectbox", label_visibility="collapsed"
    )
    process_query_button = st.button("Ask this question", key="query_button")

    # Access loaded models
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_role = None
    for message in st.session_state.chat_history:
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    if process_query_button:
        if selected_query == "Choose your question":
            st.error("⚠️ Please select your question from the dropdown.")
        else:
            prompt_from_dropdown = selected_query[0].upper() + selected_query[1:]

            st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
            if last_role == "assistant":
                st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt_from_dropdown, unsafe_allow_html=True)
            last_role = "user"

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""

                if is_ood(prompt_from_dropdown, clf_model, clf_tokenizer):
                    full_response = random.choice(fallback_responses)
                else:
                    with st.spinner("Generating response..."):
                        dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp)
                        response_gpt = generate_response(model, tokenizer, prompt_from_dropdown)
                        full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)

                streamed_text = ""
                for word in full_response.split():
                    streamed_text += word + " "
                    message_placeholder.markdown(streamed_text + "▌", unsafe_allow_html=True)
                    time.sleep(0.05)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"
            st.rerun()

    if prompt := st.chat_input("Enter your own question:"):
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
        if not prompt.strip():
            st.toast("⚠️ Please enter a question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
            if last_role == "assistant":
                st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt, unsafe_allow_html=True)
            last_role = "user"

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""

                if is_ood(prompt, clf_model, clf_tokenizer):
                    full_response = random.choice(fallback_responses)
                else:
                    with st.spinner("Generating response..."):
                        dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                        response_gpt = generate_response(model, tokenizer, prompt)
                        full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)

                streamed_text = ""
                for word in full_response.split():
                    streamed_text += word + " "
                    message_placeholder.markdown(streamed_text + "▌", unsafe_allow_html=True)
                    time.sleep(0.05)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"
            st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Chat", key="reset_button"):
            st.session_state.chat_history = []
            last_role = None
            st.rerun()

# ==============================================================
# DISCLAIMER (appears at the very bottom of the page)
# ==============================================================

st.markdown(
    """
<p style='font-size:0.85em; color:#666; text-align:center; margin-top:2rem;'>
This is not a conversational AI. It is designed solely for event ticketing queries. 
Responses outside this scope may be inaccurate.
</p>
""",
    unsafe_allow_html=True
)
