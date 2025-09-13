import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification
)
import spacy
import time
import random

# =============================
# MODEL AND CONFIGURATION SETUP
# =============================

# Hugging Face model IDs
GPT2_MODEL_ID = "Zlib2/ETCSCb_DistilGPT2"
CLASSIFIER_ID = "Zlib2/Query_Classifier_DistilBERT"

# Random OOD Fallback Responses
fallback_responses = [
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    # ... (rest of the fallback responses remain the same)
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

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier model from Hugging Face Hub. Error: {e}")
        return None, None

def is_ood(query: str, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return pred_id == 1

# =============================
# ORIGINAL HELPER FUNCTIONS
# =============================

static_placeholders = {
    # ... (static placeholders remain the same)
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
# CSS AND UI SETUP
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

/* --- CORRECTED FOOTER STYLING --- */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: var(--streamlit-background-color);
    color: gray;
    text-align: center;
    padding: 5px 0;
    font-size: 13px;
    z-index: 9999;
}
.main { padding-bottom: 40px; }
</style>
    """, unsafe_allow_html=True
)

# Add the fixed footer
st.markdown(
    """
    <div class="footer">
        This is not a conversational AI. It is designed solely for <b>event ticketing</b> queries. Responses outside this scope may be inaccurate.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='font-size: 43px;'>Advanced Event Ticketing Chatbot</h1>", unsafe_allow_html=True)

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

example_queries = [
    "How do I buy a ticket?", "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?", "How can I find details about upcoming events?",
    "How do I contact customer service?", "How do I get a refund?", "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?", "How can I sell my ticket?"
]

# Load models
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

# ==================================
# MAIN CHAT INTERFACE
# ==================================

if st.session_state.models_loaded:
    st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")

    selected_query = st.selectbox(
        "Choose a query from examples:", ["Choose your question"] + example_queries,
        key="query_selectbox", label_visibility="collapsed"
    )
    process_query_button = st.button("Ask this question", key="query_button")

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
        elif selected_query:
            prompt_from_dropdown = selected_query
            prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

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
                for word in full_response.split(" "):
                    streamed_text += word + " "
                    message_placeholder.markdown(streamed_text + "⬤", unsafe_allow_html=True)
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
                for word in full_response.split(" "):
                    streamed_text += word + " "
                    message_placeholder.markdown(streamed_text + "⬤", unsafe_allow_html=True)
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
