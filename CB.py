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

GPT2_MODEL_ID = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"  

fallback_responses = [
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets.",
    "Apologies, but this falls outside the scope of my support. I’m here if you need any help with event ticket issues.",
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
    return pred_id == 1  # OOD if label 1

# =============================
# HELPER FUNCTIONS
# =============================

static_placeholders = {"{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)"}

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
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# =============================
# CSS AND UI
# =============================

st.markdown(
    """
<style>
.stButton>button { background: linear-gradient(90deg, #ff8a00, #e52e71); color: white !important; border: none; border-radius: 25px; padding: 10px 20px; }
.footer {position: fixed;left:0;bottom:0;width:100%;background: var(--streamlit-background-color);color: gray;text-align:center;padding:5px;font-size:13px;}
.main { padding-bottom: 40px; }
.response-time { font-size: 12px; color: gray; margin-top: -8px; }
</style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        This is not a conversational AI. It is designed solely for <b>event ticketing</b> queries. Responses outside this scope may be inaccurate.
    </div>
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
    "How do I change my personal details on my ticket?", "How can I find details about upcoming events?"
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

    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_role = None
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message["role"] == "assistant" and "time_taken" in message:
                st.markdown(f"<p class='response-time'>⏱️ Response generated in {message['time_taken']:.2f} seconds</p>", unsafe_allow_html=True)

    def process_prompt(prompt):
        start_time = time.time()
        if is_ood(prompt, clf_model, clf_tokenizer):
            full_response = random.choice(fallback_responses)
        else:
            with st.spinner("Generating response..."):
                dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                response_gpt = generate_response(model, tokenizer, prompt)
                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
        end_time = time.time()
        elapsed = end_time - start_time

        st.session_state.chat_history.append({
            "role": "assistant", "content": full_response, "avatar": "🤖", "time_taken": elapsed
        })
        st.rerun()

    if process_query_button and selected_query != "Choose your question":
        prompt = selected_query
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        process_prompt(prompt)

    if prompt := st.chat_input("Enter your own question:"):
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        process_prompt(prompt)

    if st.session_state.chat_history:
        if st.button("Clear Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.rerun()
