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

GPT2_MODEL_ID   = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID   = "IamPradeep/Query_Classifier_DistilBERT"

# Random OOD fallback responses
fallback_responses = [
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    # … (keep the rest as-is)
]

# =============================
# MODEL-LOADING HELPERS
# =============================

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    try:
        model     = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_ID, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load GPT-2 model. Error: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    try:
        tok   = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return model, tok
    except Exception as e:
        st.error(f"Failed to load classifier. Error: {e}")
        return None, None

def is_ood(query: str, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True,
                       padding=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return torch.argmax(out.logits, dim=1).item() == 1   # label 1 == OOD

# =============================
# REPLACEMENT / GENERATION UTILS
# =============================

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    # … (keep all the other static placeholders unchanged)
}

def replace_placeholders(resp, dyn, stat):
    for k, v in stat.items():
        resp = resp.replace(k, v)
    for k, v in dyn.items():
        resp = resp.replace(k, v)
    return resp

def extract_dynamic_placeholders(user_q, nlp):
    doc = nlp(user_q)
    dyn = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            dyn['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            dyn['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
    dyn.setdefault('{{EVENT}}', "event")
    dyn.setdefault('{{CITY}}',  "city")
    return dyn

def generate_response(model, tokenizer, instruction, max_len=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inp = f"Instruction: {instruction} Response:"
    toks = tokenizer(inp, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **toks, max_length=max_len, temperature=0.6,
            top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    txt = tokenizer.decode(gen[0], skip_special_tokens=True)
    return txt.split("Response:", 1)[1].strip()

# =============================
# CSS  +  FOOTER
# =============================

st.markdown(
    """
<style>
.stButton>button { background: linear-gradient(90deg,#ff8a00,#e52e71); color:white!important;
    border:none; border-radius:25px; padding:10px 20px; font-size:1.2em; font-weight:bold;
    cursor:pointer; transition:transform .2s ease, box-shadow .2s ease; margin-top:5px;}
.stButton>button:hover { transform:scale(1.05); box-shadow:0 5px 15px rgba(0,0,0,.3);}
.stButton>button:active{ transform:scale(.98);}
*{font-family:'Times New Roman',Times,serif!important;}
.horizontal-line{border-top:2px solid #e0e0e0; margin:15px 0;}
div[data-testid="stChatInput"]{box-shadow:0 4px 8px rgba(0,0,0,.2); border-radius:5px; padding:10px; margin:10px 0;}
.footer{position:fixed;left:0;bottom:0;width:100%;background:var(--streamlit-background-color);
    color:gray;text-align:center;padding:5px 0;font-size:13px;z-index:9999;}
.main{padding-bottom:40px;}
</style>""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="footer">
    This is not a conversational AI. It is designed solely for <b>event ticketing</b> queries.
    Responses outside this scope may be inaccurate.
</div>
""",
    unsafe_allow_html=True
)

st.markdown("<h1 style='font-size:43px;'>Advanced Event Ticketing Chatbot</h1>", unsafe_allow_html=True)

# =============================
# INITIAL MODEL LOAD
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources…"):
        try:
            nlp                  = load_spacy_model()
            gpt2_model, gpt2_tok = load_gpt2_model_and_tokenizer()
            clf_model, clf_tok   = load_classifier_model()
            if all([nlp, gpt2_model, gpt2_tok, clf_model, clf_tok]):
                st.session_state.update({
                    "models_loaded": True,
                    "nlp": nlp,
                    "model": gpt2_model,
                    "tokenizer": gpt2_tok,
                    "clf_model": clf_model,
                    "clf_tokenizer": clf_tok
                })
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error loading models: {e}")

# =============================
# MAIN CHAT INTERFACE
# =============================

if st.session_state.models_loaded:

    example_queries = [
        "How do I buy a ticket?", "How can I upgrade my ticket for the upcoming event in Hyderabad?",
        "How do I change my personal details on my ticket?", "How can I find details about upcoming events?",
        "How do I contact customer service?", "How do I get a refund?", "What is the ticket cancellation fee?",
        "How can I track my ticket cancellation status?", "How can I sell my ticket?"
    ]

    st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")

    sel = st.selectbox(
        "Choose a query:", ["Choose your question"] + example_queries,
        key="query_selectbox", label_visibility="collapsed"
    )
    if st.button("Ask this question", key="query_button"):
        if sel == "Choose your question":
            st.error("⚠️ Please select a question from the dropdown.")
        else:
            st.session_state["new_prompt"] = sel

    # Retrieve possible manual text input
    user_input = st.chat_input("Enter your own question:")

    if user_input:
        st.session_state["new_prompt"] = user_input

    # Display previous chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_role = None
    for m in st.session_state.chat_history:
        if m["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(m["role"], avatar=m["avatar"]):
            st.markdown(m["content"], unsafe_allow_html=True)
        last_role = m["role"]

    # =============================
    # HANDLE A NEW PROMPT (if any)
    # =============================
    if st.session_state.get("new_prompt"):
        prompt = st.session_state.pop("new_prompt")
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

        # ----- 1. show user message -----
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        if last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        last_role = "user"

        # ----- 2. assistant reply (with timing) -----
        with st.chat_message("assistant", avatar="🤖"):
            msg_placeholder   = st.empty()
            time_placeholder  = st.empty()   # new
            start_time        = time.perf_counter()  # start stopwatch

            # Decide response
            if is_ood(prompt,
                      st.session_state.clf_model,
                      st.session_state.clf_tokenizer):
                full_resp = random.choice(fallback_responses)
            else:
                with st.spinner("Generating response..."):
                    dyn   = extract_dynamic_placeholders(prompt, st.session_state.nlp)
                    gpt_r = generate_response(st.session_state.model,
                                              st.session_state.tokenizer,
                                              prompt)
                    full_resp = replace_placeholders(gpt_r, dyn, static_placeholders)

            # typing-like streaming
            streamed = ""
            for word in full_resp.split():
                streamed += word + " "
                msg_placeholder.markdown(streamed + "⬤", unsafe_allow_html=True)
                time.sleep(0.05)
            msg_placeholder.markdown(full_resp, unsafe_allow_html=True)

            # stop stopwatch & render time
            elapsed = time.perf_counter() - start_time
            time_placeholder.markdown(
                f"<span style='color:gray;font-size:0.85em;'>⏱️ Response time: {elapsed:.2f} s</span>",
                unsafe_allow_html=True
            )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_resp, "avatar": "🤖"}
        )
        last_role = "assistant"
        st.experimental_rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.experimental_rerun()
