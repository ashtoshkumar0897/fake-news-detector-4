import streamlit as st
import numpy as np
import re
import string
import io
import os
import base64
import json
import time
import tempfile
import requests
from PIL import Image, ImageChops
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Multimodal Fake News & Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #080810; color: #e0e0f0; }

    .hero {
        text-align: center; padding: 2.5rem 0 0.5rem 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(124,106,247,0.13) 0%, transparent 70%);
    }
    .hero h1 {
        font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem; letter-spacing: -1.5px;
        background: linear-gradient(135deg, #ffffff 0%, #a89cf7 35%, #7c6af7 65%, #06b6d4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        margin-bottom: 0.4rem;
    }
    .hero p { color: #5a5a7a; font-size: 0.95rem; font-weight: 300; }

    .pill-row { text-align: center; margin: 0.7rem 0 1.5rem 0; }
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(124,106,247,0.08); border: 1px solid rgba(124,106,247,0.22);
        border-radius: 20px; padding: 3px 12px; font-size: 0.73rem; color: #9a8cef;
        margin: 2px 3px; font-weight: 500; letter-spacing: 0.3px;
    }
    .pill-green  { background: rgba(52,211,153,0.08);  border-color: rgba(52,211,153,0.25);  color: #6ee7b7; }
    .pill-cyan   { background: rgba(6,182,212,0.08);   border-color: rgba(6,182,212,0.25);   color: #67e8f9; }
    .pill-orange { background: rgba(251,146,60,0.08);  border-color: rgba(251,146,60,0.25);  color: #fdba74; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: transparent; border-bottom: 1px solid #1a1a2e; padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: #0f0f1a; border: 1px solid #1e1e32; border-bottom: none;
        border-radius: 8px 8px 0 0; padding: 0.5rem 1.2rem; color: #5a5a7a;
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.82rem; letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        background: #13132a !important; border-color: #7c6af7 !important; color: #a89cf7 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #0d0d1e; border: 1px solid #1a1a30; border-top: none;
        border-radius: 0 8px 8px 8px; padding: 1.8rem;
    }

    div[data-testid="stTextArea"] label, div[data-testid="stFileUploader"] label,
    div[data-testid="stTextInput"] label {
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.78rem;
        color: #7070a0; letter-spacing: 1.8px; text-transform: uppercase;
    }
    div[data-testid="stTextArea"] textarea {
        background: #0a0a16 !important; border: 1px solid #1e1e34 !important;
        border-radius: 10px !important; color: #e0e0f0 !important;
        font-size: 0.92rem !important; line-height: 1.7 !important; padding: 14px !important;
    }
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #7c6af7 !important; box-shadow: 0 0 0 2px rgba(124,106,247,0.12) !important;
    }
    div[data-testid="stFileUploader"] {
        background: #0a0a16; border: 1px dashed #2a2a44; border-radius: 10px; padding: 0.8rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c6af7, #5a4cd4);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 1.8rem;
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.88rem;
        letter-spacing: 0.8px; transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8f7ef9, #6b5de8);
        transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,106,247,0.3);
    }

    .result-card { border-radius: 14px; padding: 1.8rem; text-align: center; margin-top: 1rem; }
    .result-fake    { background: linear-gradient(135deg,rgba(239,68,68,0.07),rgba(220,38,38,0.03));    border: 1px solid rgba(239,68,68,0.22); }
    .result-real    { background: linear-gradient(135deg,rgba(52,211,153,0.07),rgba(16,185,129,0.03));  border: 1px solid rgba(52,211,153,0.22); }
    .result-warning { background: linear-gradient(135deg,rgba(251,191,36,0.07),rgba(245,158,11,0.03)); border: 1px solid rgba(251,191,36,0.22); }
    .result-neutral { background: rgba(124,106,247,0.05); border: 1px solid rgba(124,106,247,0.18); }
    .result-ai      { background: linear-gradient(135deg,rgba(6,182,212,0.07),rgba(124,106,247,0.05)); border: 1px solid rgba(6,182,212,0.25); }

    .verdict-label  { font-family:'Syne',sans-serif; font-size:0.72rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; margin-bottom:0.4rem; }
    .verdict-text   { font-family:'Syne',sans-serif; font-weight:800; font-size:2.2rem; letter-spacing:-1px; line-height:1; margin-bottom:0.2rem; }
    .verdict-emoji  { font-size:1.9rem; margin-bottom:0.7rem; }

    .confidence-section { margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.05); }
    .confidence-label   { font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#5a5a7a; margin-bottom:0.4rem; }
    .confidence-value   { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; color:#e0e0f0; }

    .progress-bar-bg { background:rgba(255,255,255,0.05); border-radius:100px; height:5px; margin-top:0.5rem; overflow:hidden; }
    .bar-fake   { height:100%; border-radius:100px; background:linear-gradient(90deg,#ef4444,#f97316); }
    .bar-real   { height:100%; border-radius:100px; background:linear-gradient(90deg,#34d399,#06b6d4); }
    .bar-warn   { height:100%; border-radius:100px; background:linear-gradient(90deg,#fbbf24,#f59e0b); }
    .bar-purple { height:100%; border-radius:100px; background:linear-gradient(90deg,#7c6af7,#06b6d4); }
    .bar-cyan   { height:100%; border-radius:100px; background:linear-gradient(90deg,#06b6d4,#6366f1); }

    .metric-row   { display:flex; gap:0.8rem; margin-top:0.8rem; flex-wrap:wrap; }
    .metric-chip  {
        background:#0f0f1e; border:1px solid #1e1e34; border-radius:8px;
        padding:0.5rem 0.8rem; font-family:'DM Mono',monospace;
        font-size:0.72rem; color:#8080b0; flex:1; min-width:110px; text-align:center;
    }
    .metric-chip strong { color:#c0c0e0; display:block; font-size:0.88rem; }
    .metric-chip-ai     { background:rgba(6,182,212,0.06); border:1px solid rgba(6,182,212,0.2); }
    .metric-chip-ai strong { color:#67e8f9; }
    .metric-chip-bert   { background:rgba(124,106,247,0.06); border:1px solid rgba(124,106,247,0.2); }
    .metric-chip-bert strong { color:#a89cf7; }

    .api-badge {
        display:inline-flex; align-items:center; gap:5px;
        background:rgba(251,146,60,0.08); border:1px solid rgba(251,146,60,0.2);
        border-radius:20px; padding:3px 10px; font-size:0.72rem; color:#fdba74;
        margin-bottom:0.6rem; font-family:'DM Mono',monospace;
    }
    .dl-badge {
        display:inline-flex; align-items:center; gap:5px;
        background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.2);
        border-radius:20px; padding:3px 10px; font-size:0.72rem; color:#67e8f9;
        margin-bottom:0.6rem; font-family:'DM Mono',monospace;
    }
    .ela-badge {
        display:inline-flex; align-items:center; gap:5px;
        background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.2);
        border-radius:20px; padding:3px 10px; font-size:0.72rem; color:#fcd34d;
        margin-bottom:0.6rem; font-family:'DM Mono',monospace;
    }

    .status-ok   { background:rgba(52,211,153,0.06); border:1px solid rgba(52,211,153,0.2); border-radius:8px; padding:0.4rem 0.9rem; font-size:0.75rem; color:#6ee7b7; display:inline-block; font-family:'DM Mono',monospace; }
    .status-warn { background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.2); border-radius:8px; padding:0.4rem 0.9rem; font-size:0.75rem; color:#fcd34d; display:inline-block; font-family:'DM Mono',monospace; }
    .status-fail { background:rgba(239,68,68,0.06);  border:1px solid rgba(239,68,68,0.2);  border-radius:8px; padding:0.4rem 0.9rem; font-size:0.75rem; color:#fca5a5; display:inline-block; font-family:'DM Mono',monospace; }

    .section-header {
        font-family:'Syne',sans-serif; font-weight:700; font-size:0.78rem;
        letter-spacing:1.8px; text-transform:uppercase; color:#5a5a8a;
        margin:1.2rem 0 0.6rem 0; padding-bottom:0.4rem; border-bottom:1px solid #1a1a2e;
    }
    .combined-card {
        background:linear-gradient(135deg,rgba(124,106,247,0.06),rgba(6,182,212,0.04));
        border:1px solid rgba(124,106,247,0.2); border-radius:16px; padding:2rem; margin-top:2rem;
    }
    .combined-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:#a89cf7; margin-bottom:1.2rem; }
    .modal-row      { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:0.8rem; margin-bottom:1rem; }
    .modal-tile     { background:#0a0a16; border:1px solid #1e1e34; border-radius:10px; padding:0.9rem; text-align:center; }
    .modal-tile-label  { font-size:0.68rem; letter-spacing:1.5px; text-transform:uppercase; color:#44447a; margin-bottom:0.4rem; }
    .modal-tile-score  { font-family:'Syne',sans-serif; font-weight:800; font-size:1.5rem; margin-bottom:0.1rem; }
    .modal-tile-verdict { font-size:0.72rem; color:#6a6a9a; }

    .info-box  { background:rgba(6,182,212,0.05); border:1px solid rgba(6,182,212,0.18); border-radius:10px; padding:1rem 1.2rem; font-size:0.84rem; color:#67e8f9; margin-top:0.8rem; line-height:1.6; }
    .warn-box  { background:rgba(251,191,36,0.05); border:1px solid rgba(251,191,36,0.18); border-radius:10px; padding:1rem 1.2rem; font-size:0.84rem; color:#fcd34d; margin-top:0.8rem; line-height:1.6; }
    .error-box { background:rgba(239,68,68,0.05);  border:1px solid rgba(239,68,68,0.18);  border-radius:10px; padding:1rem 1.2rem; font-size:0.84rem; color:#fca5a5; margin-top:0.8rem; }
    .fallback-box { background:rgba(251,146,60,0.05); border:1px solid rgba(251,146,60,0.2); border-radius:10px; padding:1rem 1.2rem; font-size:0.84rem; color:#fdba74; margin-top:0.8rem; line-height:1.6; }

    .similarity-meter {
        background:#0a0a16; border:1px solid #1e1e34; border-radius:12px;
        padding:1.4rem; margin-top:1rem; text-align:center;
    }
    .similarity-value { font-family:'Syne',sans-serif; font-weight:800; font-size:3rem; letter-spacing:-2px; }
    .similarity-label { font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; color:#5a5a7a; }

    .divider { border:none; border-top:1px solid #14142a; margin:1.8rem 0; }
    footer{visibility:hidden;} #MainMenu{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


HF_API_BASE     = "https://api-inference.huggingface.co/models"
MODEL_TEXT      = "vikram71198/distilroberta-base-finetuned-fake-news-detection"
MODEL_AI_IMAGE  = "prithivMLmods/AI-vs-Deepfake-vs-Real-9999"
MODEL_CLIP      = "openai/clip-vit-base-patch32"
API_TIMEOUT     = 30
API_MAX_RETRIES = 3


def get_hf_key():
    try:
        return st.secrets["HF_API_KEY"]
    except Exception:
        return os.environ.get("HF_API_KEY", "")


def hf_headers(key):
    return {"Authorization": f"Bearer {key}"}


def hf_post_json(endpoint, payload, key, retries=API_MAX_RETRIES):
    url     = f"{HF_API_BASE}/{endpoint}"
    headers = {**hf_headers(key), "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
            if resp.status_code in (410, 401, 403):
                return None, f"HTTP {resp.status_code} — model removed or unauthorised. Check model name / API key."
            if resp.status_code == 503:
                wait = min(8 * (attempt + 1), 25)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json(), None
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.Timeout:
            last_err = "Request timed out"
        except requests.exceptions.ConnectionError as exc:
            last_err = f"Connection error: {exc}"
        except Exception as exc:
            last_err = str(exc)
    return None, last_err or "Max retries exceeded"


def compress_image_for_api(img_pil, max_side=320, max_bytes=90000, quality_start=82):
    img = img_pil.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    for q in range(quality_start, 38, -8):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=38)
    return buf.getvalue()


def hf_post_bytes(endpoint, image_bytes, key, retries=API_MAX_RETRIES):
    url      = f"{HF_API_BASE}/{endpoint}"
    headers  = {**hf_headers(key), "Content-Type": "application/octet-stream"}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                url, headers=headers, data=image_bytes,
                timeout=API_TIMEOUT, stream=False
            )
            if resp.status_code in (410, 401, 403):
                return None, f"HTTP {resp.status_code} — model removed or unauthorised. Check model name / API key."
            if resp.status_code == 503:
                time.sleep(min(8 * (attempt + 1), 25))
                continue
            if resp.status_code == 200:
                try:
                    return resp.json(), None
                except Exception:
                    return None, "JSON decode error"
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.ChunkedEncodingError as exc:
            last_err = f"IncompleteRead — image too large or connection dropped: {str(exc)[:120]}"
            time.sleep(4 * (attempt + 1))
        except requests.exceptions.Timeout:
            last_err = "Request timed out"
        except requests.exceptions.ConnectionError as exc:
            last_err = f"Connection error: {str(exc)[:120]}"
        except Exception as exc:
            last_err = str(exc)[:200]
    return None, last_err or "Max retries exceeded"


def pil_to_bytes(img_pil, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    img_pil.convert("RGB").save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def pil_to_b64(img_pil, fmt="JPEG", quality=90):
    return base64.b64encode(pil_to_bytes(img_pil, fmt, quality)).decode("utf-8")


def api_predict_text_fake_news(text, key):
    truncated  = text[:1800]
    result, err = hf_post_json(MODEL_TEXT, {"inputs": truncated}, key)
    if err or result is None:
        return None, None, err

    scores = result[0] if isinstance(result[0], list) else result

    fake_score = 0.0
    real_score = 0.0
    for item in scores:
        lbl = item.get("label", "").lower()
        s   = float(item.get("score", 0))
        if "not_fake" in lbl or "real" in lbl:
            real_score = s
        elif "fake" in lbl:
            fake_score = s

    if fake_score == 0.0 and real_score == 0.0 and scores:
        fake_score = float(scores[0].get("score", 0.5))
        real_score = 1.0 - fake_score

    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = fake_score if label == "FAKE" else real_score
    return label, round(confidence * 100, 1), None


def api_detect_ai_image(img_pil, key, max_side=320):
    img_bytes   = compress_image_for_api(img_pil, max_side=max_side)
    result, err = hf_post_bytes(MODEL_AI_IMAGE, img_bytes, key)
    if err or result is None:
        return None, err

    real_score = 0.0
    fake_score = 0.0
    for item in result:
        lbl = item.get("label", "").lower()
        s   = float(item.get("score", 0.0))
        if "real" in lbl:
            real_score = s
        else:
            fake_score += s
    if real_score == 0.0 and fake_score == 0.0:
        return 50.0, None
    total = real_score + fake_score
    return round((fake_score / total) * 100, 1) if total > 0 else 50.0, None


def api_clip_similarity(img_pil, headline_text, key, max_side=224):
    img_bytes   = compress_image_for_api(img_pil, max_side=max_side)
    candidate_a = headline_text[:200]
    candidate_b = "an unrelated image with different subject matter"

    result, err = hf_post_bytes(MODEL_CLIP, img_bytes, key)
    if err or result is None:
        small_b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {
            "inputs": {"image": small_b64},
            "parameters": {"candidate_labels": [candidate_a, candidate_b]},
        }
        result, err = hf_post_json(MODEL_CLIP, payload, key)

    if err or result is None:
        return None, None, err

    items = result if isinstance(result, list) else result.get("outputs", [])
    match_score = 0.5
    for item in items:
        lbl = item.get("label", "")
        if lbl == candidate_a or (candidate_a[:30] in lbl):
            match_score = float(item.get("score", 0.5))
            break

    similarity_pct = round(match_score * 100, 1)
    mismatch_pct   = round(100 - similarity_pct, 1)
    return similarity_pct, mismatch_pct, None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_fallback_training_data():
    real_samples = [
        "The Federal Reserve raised interest rates by a quarter point citing progress on inflation and a resilient labor market. Officials said they would remain data-dependent in future decisions.",
        "Scientists at NASA announced the successful launch of the Artemis mission marking a significant step toward returning humans to the Moon after two delays due to weather conditions.",
        "The World Health Organization declared an end to the COVID-19 global health emergency but warned the virus remains a persistent public health threat requiring ongoing surveillance.",
        "Google unveiled its latest artificial intelligence model at its developer conference demonstrating new capabilities in multimodal reasoning code generation and language translation.",
        "The United Nations Security Council passed a resolution calling for an immediate ceasefire with fourteen members voting in favor after weeks of diplomatic negotiations.",
        "Apple reported record quarterly earnings driven by strong iPhone sales in emerging markets and continued growth in its services division which now exceeds a quarter of total revenue.",
        "Researchers at Johns Hopkins University published findings showing a new drug compound significantly reduced tumor growth in patients with advanced pancreatic cancer.",
        "The European Parliament voted to approve the landmark AI Act the world first comprehensive legal framework for artificial intelligence classifying systems by risk level.",
        "India successfully landed the Chandrayaan spacecraft near the lunar south pole becoming the first country to achieve a soft landing in that region believed to contain water ice.",
        "The International Monetary Fund revised its global growth forecast upward citing stronger performance in the United States India and several emerging market economies.",
        "Senate lawmakers reached a bipartisan agreement on a supplemental funding package after months of difficult negotiations threatening support for key government programs.",
        "A study published in Nature found global sea levels rose at an accelerated rate last year driven by melting ice sheets in Greenland and Antarctica losing mass faster than projected.",
        "Microsoft completed its major acquisition after receiving regulatory approval concluding one of the largest mergers in technology history after extended legal proceedings.",
        "Thousands of automobile workers went on strike after contract negotiations broke down over wages benefits and the transition to electric vehicle production raising supply chain concerns.",
        "The Bank of England held interest rates steady signaling policymakers believe monetary tightening is beginning to cool inflation without triggering a severe economic downturn.",
        "A bipartisan group of senators introduced legislation to strengthen data privacy protections requiring companies to obtain explicit consent before collecting sensitive personal information.",
        "Tesla reported quarterly delivery numbers fell short of analyst expectations amid concern about demand softening and increased competition from domestic and foreign electric vehicle makers.",
        "Wildfires burning across western North America forced evacuations of thousands of residents and caused air quality to deteriorate to hazardous levels in major cities across the region.",
        "Harvard University announced a new financial aid policy making tuition free for undergraduate students from families earning below a threshold improving campus economic diversity.",
        "Congress passed a bipartisan infrastructure bill allocating over one trillion dollars for roads bridges broadband internet expansion and public transit improvements over eight years.",
        "The Securities and Exchange Commission charged several cryptocurrency exchanges with operating as unregistered securities dealers escalating federal regulatory scrutiny of digital assets.",
        "A federal judge blocked a major administration policy ruling the executive branch exceeded its authority in attempting to implement the measure without congressional authorization.",
        "The pharmaceutical company announced positive results from its phase three clinical trial for a treatment targeting drug-resistant tuberculosis showing high success across thousands of patients.",
        "A peer-reviewed study found that regular moderate exercise reduces the risk of developing type two diabetes by nearly forty percent even among individuals with genetic predisposition.",
        "State election officials confirmed all voting machines passed pre-election logic and accuracy tests and results were verified through a mandatory post-election audit required under state law.",
    ]
    fake_samples = [
        "BREAKING Scientists confirm drinking bleach mixed with lemon juice cures any disease The government has been hiding this miracle cure for decades to protect pharmaceutical profits Share before deleted",
        "EXCLUSIVE The moon landing was staged in a Hollywood studio Newly leaked documents from a whistleblower prove the entire Apollo program was an elaborate hoax to win the space race",
        "URGENT Bill Gates has been secretly microchipping millions through COVID vaccines The chips activate when exposed to signals and allow global elites to control your thoughts and movements",
        "SHOCKING A former president was born in a foreign country A classified document has confirmed this massive cover-up that the mainstream media and deep state refuse to report",
        "ALERT The government is putting fluoride in the water supply to make people docile An anonymous insider leaked evidence proving this is a decades-long mind control program",
        "MIRACLE CURE A simple mixture of apple cider vinegar and baking soda cures cancer diabetes and heart disease Big Pharma is suppressing this information to protect their trillion dollar profits",
        "EXPOSED Top politicians run a child trafficking ring Coded emails contain irrefutable proof of this satanic operation involving powerful officials and the mainstream media is covering it up",
        "PROOF The Earth is actually flat and space agencies have been lying for centuries Leaked documents show astronauts are paid actors using green screens and CGI to deceive the public",
        "BOMBSHELL A recent election was stolen by millions of fraudulent ballots Forensic audits uncovered the true count being suppressed by corrupt media and political establishment traitors",
        "REVEALED Chemtrails sprayed by military aircraft contain mind-altering chemicals making the population sick A retired pilot has come forward with photographic proof of this secret program",
        "WARNING The COVID vaccine contains nanoparticles that cause infertility and destroy the immune system Thousands of doctors are being silenced and censored for trying to warn the public",
        "CONFIRMED A billionaire is personally funding a migrant invasion by paying thousands of dollars for people to cross the border illegally and vote for specific political candidates in elections",
        "SHOCKING DISCOVERY Ancient pyramids were built by aliens using advanced technology modern humans cannot replicate Mainstream archaeologists have been hiding this evidence for decades",
        "EXPLOSIVE A secret elite group plans to reduce the global population using engineered pandemics chemtrails and tainted water supplies They are executing this depopulation agenda right now",
        "WATCH A top health official admitted on hidden camera a virus was created in a laboratory and intentionally released for political purposes The clip has been banned from all major platforms",
        "UNBELIEVABLE Scientists discover a secret organ the medical establishment has been deliberately hiding because treating it would eliminate the need for expensive medical procedures and drugs",
        "LEAKED Internal documents confirm a major vaccine was never tested for safety and causes permanent DNA damage in the vast majority of recipients but regulators were bribed to approve it",
        "ALERT The deep state is planning to cancel the next presidential election and install a permanent authoritarian regime Patriotic insiders leaked the timeline showing it begins very soon",
        "STUNNING Doctors now admit cell phones cause brain tumors in nearly all users within five years but the telecommunications industry paid billions to suppress the studies proving this fact",
        "BREAKING EXCLUSIVE The real unemployment rate is over seventy percent but government agencies falsified data for twenty years under orders from both political parties to deceive Americans",
        "URGENT WARNING A deadly disease engineered in a foreign laboratory is spreading rapidly and will kill anyone vaccinated against COVID Only natural immunity provides protection from this bioweapon",
        "EXPOSED The real reason for inflation is a secret coordinated effort by billionaires to impoverish the middle class and force them into permanent debt slavery under the New World Order",
        "SHOCKING PROOF Satellite imagery confirms massive underground cities built by the government to house elites during a planned catastrophic event designed to target ordinary citizens worldwide",
        "BOMBSHELL VIDEO Leaked footage from a global economic forum shows world leaders laughing about deliberately destroying national economies to implement their depopulation and control agenda",
        "CONFIRMED Major elections are controlled by a supercomputer algorithm switching votes in real time during counting This is why official results always differ from exit polls patriots know",
    ]
    return real_samples + fake_samples, [1] * len(real_samples) + [0] * len(fake_samples)


@st.cache_resource(show_spinner=False)
def load_local_fallback_model():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    model_dir = "model_artifacts"
    paths = {k: os.path.join(model_dir, v) for k, v in {
        'tfidf': 'tfidf_vectorizer.pkl', 'iso': 'isolation_forest.pkl',
        'rf': 'rf_classifier.pkl',       'meta': 'meta.pkl',
    }.items()}

    if all(os.path.exists(p) for p in paths.values()):
        return (joblib.load(paths['tfidf']), joblib.load(paths['iso']),
                joblib.load(paths['rf']),    joblib.load(paths['meta']))

    texts, labels = get_fallback_training_data()
    cleaned = [clean_text(t) for t in texts]

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_t   = tfidf.fit_transform(cleaned)
    iso   = IsolationForest(n_estimators=50, contamination=0.15, random_state=42)
    iso.fit(X_t)
    anm   = iso.decision_function(X_t).reshape(-1, 1)
    X_h   = np.hstack([X_t.toarray(), anm])
    y     = np.array(labels)
    Xtr, Xte, ytr, yte = train_test_split(X_h, y, test_size=0.2, random_state=42, stratify=y)
    rf    = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(Xtr, ytr)
    acc   = accuracy_score(yte, rf.predict(Xte))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(tfidf, paths['tfidf']); joblib.dump(iso, paths['iso'])
    joblib.dump(rf,    paths['rf'])
    joblib.dump({"accuracy": float(acc), "train_size": len(Xtr),
                 "n_features": X_h.shape[1], "source": "built-in"}, paths['meta'])
    return tfidf, iso, rf, joblib.load(paths['meta'])


def local_predict_text(text, tfidf, iso, rf):
    c   = clean_text(text)
    v   = tfidf.transform([c])
    a   = iso.decision_function(v).reshape(-1, 1)
    h   = np.hstack([v.toarray(), a])
    p   = rf.predict(h)[0]
    prb = rf.predict_proba(h)[0]
    lbl = "REAL" if p == 1 else "FAKE"
    return lbl, round(float(prb[p]) * 100, 1)


def resize_for_analysis(img_pil, max_dim=900):
    w, h = img_pil.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        return img_pil.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return img_pil


def perform_ela(img_pil, quality=75, amplify=18):
    rgb  = img_pil.convert('RGB')
    buf  = io.BytesIO()
    rgb.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    rc   = Image.open(buf).convert('RGB')
    diff = ImageChops.difference(rgb, rc)
    arr  = np.array(diff, dtype=np.float32)
    m, s = float(np.mean(arr)), float(np.std(arr))
    sp   = float(np.mean(arr > (m + 2.5 * s)) * 100)
    amp  = np.clip(arr * amplify, 0, 255).astype(np.uint8)
    return {'ela_array': amp, 'mean_error': m, 'std_error': s,
            'max_error': float(np.max(arr)), 'suspicious_pct': sp}


def analyze_noise_consistency(img_pil, block_size=32):
    gray = np.array(img_pil.convert('L'), dtype=np.uint8)
    h, w = gray.shape
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    variances = [float(np.var(lap[y:y+block_size, x:x+block_size]))
                 for y in range(0, h - block_size, block_size)
                 for x in range(0, w - block_size, block_size)]
    if not variances:
        return {'inconsistency': 0.0, 'cv': 0.0}
    arr = np.array(variances)
    cv  = float(np.std(arr)) / (float(np.mean(arr)) + 1e-10)
    return {'inconsistency': float(min(cv / 4.0, 1.0)), 'cv': float(cv)}


def extract_exif_flags(img_pil):
    editing_kw = ['photoshop','gimp','lightroom','affinity','canva',
                  'midjourney','dall','stable diffusion','firefly','adobe']
    try:
        exif = img_pil._getexif()
        if exif is None:
            return {'has_editing_software': False, 'software': 'No EXIF'}
        sw  = str(exif.get(305, '')).lower()
        has = any(k in sw for k in editing_kw)
        return {'has_editing_software': has, 'software': str(exif.get(305, 'Unknown'))}
    except Exception:
        return {'has_editing_software': False, 'software': 'EXIF unreadable'}


def compute_ela_classical_score(ela, noise, exif):
    s  = min(ela['mean_error'] * 6.5, 33)
    s += min(ela['suspicious_pct'] * 0.55, 24)
    s += min(noise['inconsistency'] * 28, 28)
    s += 15 if exif['has_editing_software'] else 0
    return round(min(s, 100), 1)


def compute_image_forgery_score(classical, ai_prob=None):
    if ai_prob is None:
        return classical
    combined = classical * 0.30 + ai_prob * 0.70
    if ai_prob >= 70:
        combined = max(combined, 70)
    elif ai_prob >= 50:
        combined = max(combined, 50)
    return round(min(combined, 100), 1)


def build_ela_figure(ela_array, original_pil):
    gray_ela = np.mean(ela_array, axis=2) if ela_array.ndim == 3 else ela_array
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor('#0d0d1e')
    axes[0].imshow(original_pil); axes[0].set_title('Original', color='#c0c0e0', fontsize=9, pad=6); axes[0].axis('off')
    im = axes[1].imshow(gray_ela, cmap='inferno', vmin=0, vmax=255)
    axes[1].set_title('ELA Heatmap  ·  bright = high compression error = possible manipulation', color='#c0c0e0', fontsize=8, pad=6)
    axes[1].axis('off')
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors='#8080b0', labelsize=7); cb.set_label('Error Intensity', color='#8080b0', fontsize=7)
    fig.tight_layout(pad=1.2)
    buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor='#0d0d1e', bbox_inches='tight', dpi=110); buf.seek(0)
    plt.close(fig); return Image.open(buf)


def extract_video_frames(video_bytes, max_frames=12):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes); tmp_path = tmp.name
    frames, meta = [], {}
    try:
        cap    = cv2.VideoCapture(tmp_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        dur    = total / fps if fps > 0 else 0
        meta   = {'total_frames': total, 'fps': round(fps, 2), 'duration_s': round(dur, 2)}
        if total > 0:
            for idx in np.linspace(0, total - 1, min(max_frames, total), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.resize(frame, (480, 270)))
        cap.release()
    except Exception:
        pass
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass
    return frames, meta


@st.cache_resource(show_spinner=False)
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(frame_bgr):
    cascade = get_face_cascade()
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(28, 28))
    return [{'bbox': (x, y, w, h)} for (x, y, w, h) in faces]


def frame_ela_score(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ela = perform_ela(Image.fromarray(rgb))
    return ela['mean_error'], ela['suspicious_pct']


def temporal_inconsistency(frames):
    if len(frames) < 2:
        return 0.0, []
    diffs = [float(np.mean(np.abs(
        cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32) -
        cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY).astype(np.float32)
    ))) for i in range(1, len(frames))]
    arr    = np.array(diffs)
    mean_d = float(np.mean(arr)); std_d = float(np.std(arr))
    spikes = int(np.sum(arr > mean_d + 2.2 * std_d))
    score  = float(min(spikes / max(len(diffs), 1) * 0.6 + std_d / (mean_d + 1e-8) * 0.08, 1.0))
    return score, diffs


def skin_tone_consistency(frames, face_data_list):
    hues = []
    for frame, faces in zip(frames, face_data_list):
        for face in faces:
            x, y, w, h = face['bbox']
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue
            hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 28, 55]), np.array([25, 185, 255]))
            skin = hsv[mask > 0]
            if len(skin) > 15:
                hues.append(float(np.mean(skin[:, 0])))
    if len(hues) < 2: return 0.4
    cv = float(np.std(hues)) / (float(np.mean(hues)) + 1e-8)
    return float(min(cv * 2.8, 1.0))


def analyze_video_frames_api(frames, key, max_ai_frames=5):
    if not frames: return None, None
    step   = max(1, len(frames) // max_ai_frames)
    sample = frames[::step][:max_ai_frames]
    scores, err_last = [], None
    for frame in sample:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        p, e = api_detect_ai_image(pil, key)
        if p is not None:
            scores.append(p)
        else:
            err_last = e
    if not scores:
        return None, err_last
    return round(float(np.mean(scores)), 1), None


def compute_deepfake_score(ela_scores, temp_score, skin_score, total_faces, ai_prob=None):
    mean_ela  = float(np.mean([s[0] for s in ela_scores])) if ela_scores else 0
    classical = min(mean_ela * 3.5, 18) + min(temp_score * 22, 22) + min(skin_score * 12, 12) + (6 if total_faces == 0 else 0)
    if ai_prob is None:
        return round(min(classical, 100), 1), classical
    combined = classical * 0.30 + ai_prob * 0.70
    if ai_prob >= 50:  combined = max(combined, 65)
    elif ai_prob >= 35: combined = max(combined, 40)
    return round(min(combined, 100), 1), classical


def annotate_frame(frame_bgr, faces):
    out = frame_bgr.copy()
    for face in faces:
        x, y, w, h = face['bbox']
        cv2.rectangle(out, (x, y), (x+w, y+h), (80, 220, 140), 2)
        cv2.putText(out, 'Face', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 140), 1)
    return out


def build_frame_figure(frames, face_data_list, ela_scores, max_show=6):
    show_f = frames[:max_show]; show_fa = face_data_list[:max_show]
    n = len(show_f); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.4))
    fig.patch.set_facecolor('#0d0d1e')
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, (frame, faces) in enumerate(zip(show_f, show_fa)):
        rgb = cv2.cvtColor(annotate_frame(frame, faces), cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        ela_s = f"ELA:{ela_scores[i][0]:.2f}" if i < len(ela_scores) else ""
        axes[i].set_title(f"Frame {i+1} · {ela_s}  Faces:{len(faces)}", color='#a0a0d0', fontsize=7.5, pad=4)
        axes[i].axis('off')
    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.tight_layout(pad=0.8)
    buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor='#0d0d1e', bbox_inches='tight', dpi=100); buf.seek(0)
    plt.close(fig); return Image.open(buf)


def build_temporal_figure(diffs):
    fig, ax = plt.subplots(figsize=(7, 2.4)); fig.patch.set_facecolor('#0d0d1e'); ax.set_facecolor('#09091a')
    x = list(range(1, len(diffs) + 1))
    ax.plot(x, diffs, color='#7c6af7', linewidth=1.8, marker='o', markersize=4, zorder=3)
    ax.fill_between(x, diffs, alpha=0.18, color='#7c6af7')
    if len(diffs) > 1:
        m, s = float(np.mean(diffs)), float(np.std(diffs))
        ax.axhline(m + 2.2 * s, color='#ef4444', linewidth=1, linestyle='--', alpha=0.6, label='Spike threshold')
        ax.axhline(m,           color='#34d399', linewidth=1, linestyle='--', alpha=0.5, label='Mean')
        ax.legend(fontsize=7, facecolor='#0d0d1e', edgecolor='#2a2a44', labelcolor='#a0a0d0')
    ax.set_title('Inter-Frame Pixel Difference  ·  spikes = inconsistent transitions', color='#b0b0d0', fontsize=8, pad=6)
    ax.set_xlabel('Frame Pair', color='#7070a0', fontsize=7); ax.set_ylabel('Mean Pixel Diff', color='#7070a0', fontsize=7)
    ax.tick_params(colors='#5a5a8a', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e1e34')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', facecolor='#0d0d1e', bbox_inches='tight', dpi=100); buf.seek(0)
    plt.close(fig); return Image.open(buf)


def score_to_verdict(score, thresholds=(35, 65)):
    if score >= thresholds[1]: return "HIGH RISK", "#ef4444", "🔴"
    if score >= thresholds[0]: return "MODERATE",  "#fbbf24", "🟡"
    return "LOW RISK", "#34d399", "🟢"


def render_bar(score, bar_class):
    return (f'<div class="progress-bar-bg">'
            f'<div class="{bar_class}" style="width:{int(min(score,100))}%"></div>'
            f'</div>')


hf_key = get_hf_key()
key_ok  = bool(hf_key and len(hf_key) > 10)

st.markdown("""
<div class="hero">
    <h1>🛡️ Multimodal Fake News &amp; Deepfake Detector</h1>
    <p>API-Powered Deep Learning  ·  Zero Local ML RAM Usage  ·  4-Module Analysis</p>
</div>
<div class="pill-row">
    <span class="pill pill-orange">☁️ HF Inference API</span>
    <span class="pill pill-cyan">🤖 DistilRoBERTa Fake News</span>
    <span class="pill pill-cyan">🖼️ AI-Image Detector ViT</span>
    <span class="pill">🔬 ELA Forensics (Local)</span>
    <span class="pill pill-green">🎬 CLIP Cross-Modal Match</span>
    <span class="pill">🎥 OpenCV Frame Analysis</span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading local fallback text model…"):
    tfidf_fb, iso_fb, rf_fb, meta_fb = load_local_fallback_model()

key_status_cls  = "status-ok" if key_ok else "status-fail"
key_status_text = "✓ HF_API_KEY configured — API models active" if key_ok else "✗ HF_API_KEY not set — using local fallback models only"
acc_fb          = f"{meta_fb.get('accuracy', 0) * 100:.1f}%"
src_fb          = "CSV dataset" if meta_fb.get('source') != 'built-in' else "built-in corpus"

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-chip">Fallback Model Accuracy<strong>{acc_fb} ({src_fb})</strong></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-chip metric-chip-bert">Primary Text Model<strong>DistilRoBERTa (API)</strong></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="{key_status_cls}" style="width:100%;display:block;text-align:center;">{key_status_text}</div>', unsafe_allow_html=True)

if not key_ok:
    st.markdown("""
    <div class="warn-box" style="margin-top:0.5rem;">
        ⚙️  <strong>Setup required:</strong> Add your Hugging Face API key to unlock all DL modules.
        See the <em>Instructions</em> section at the bottom of this page.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

for k in ['text_result','image_result','video_result','cross_result']:
    if k not in st.session_state: st.session_state[k] = None

tab_text, tab_img, tab_vid, tab_cross, tab_setup = st.tabs([
    "📝  Text Analysis",
    "🖼️  Image Forensics",
    "🎥  Video Deepfake",
    "🔗  Cross-Modal CLIP",
    "⚙️  Setup & API Keys",
])


with tab_text:
    st.markdown(f"""
    <div class="info-box" style="margin-bottom:1rem;">
        <strong>Active model:</strong> {"<code>vikram71198/distilroberta-base-finetuned-fake-news-detection</code> via HF Inference API — trained on 24K+ labeled news articles" if key_ok else "Local TF-IDF + Random Forest fallback (API key not set)"}
    </div>
    """, unsafe_allow_html=True)

    news_input = st.text_area("NEWS ARTICLE / HEADLINE", placeholder="Paste any news article, headline, or social media post here…", height=200, key="text_inp")
    if st.button("ANALYZE TEXT →", key="btn_text"):
        if not news_input or len(news_input.strip().split()) < 5:
            st.markdown('<div class="error-box">✕  Please provide at least a few sentences of text.</div>', unsafe_allow_html=True)
        else:
            api_used = False
            with st.spinner("Calling DistilRoBERTa fake news API…" if key_ok else "Running local TF-IDF model…"):
                if key_ok:
                    lbl, conf, api_err = api_predict_text_fake_news(news_input, hf_key)
                    if lbl is not None:
                        api_used = True
                    else:
                        st.markdown(f'<div class="fallback-box">⚠️  API error: {api_err} — falling back to local model.</div>', unsafe_allow_html=True)
                        lbl, conf = local_predict_text(news_input, tfidf_fb, iso_fb, rf_fb)
                else:
                    lbl, conf = local_predict_text(news_input, tfidf_fb, iso_fb, rf_fb)
                    api_err   = None

            st.session_state['text_result'] = {
                'label': lbl, 'confidence': conf, 'api_used': api_used,
                'fake_pct': conf if lbl == "FAKE" else 100 - conf,
                'real_pct': conf if lbl == "REAL" else 100 - conf,
            }

    r = st.session_state['text_result']
    if r:
        lbl   = r['label']
        conf  = r['confidence']
        card  = "result-fake" if lbl == "FAKE" else "result-real"
        bar   = "bar-fake" if lbl == "FAKE" else "bar-real"
        emoji = "🚨" if lbl == "FAKE" else "✅"
        vc    = "#ef4444" if lbl == "FAKE" else "#34d399"
        model_badge = '<span class="api-badge">☁️ DistilRoBERTa via HF API</span>' if r['api_used'] else '<span class="ela-badge">💻 Local TF-IDF Fallback</span>'
        st.markdown(f"""
        <div class="result-card {card}">
            {model_badge}
            <div class="verdict-emoji">{emoji}</div>
            <div class="verdict-label">Text Verdict</div>
            <div class="verdict-text" style="color:{vc};">{lbl}</div>
            <div class="confidence-section">
                <div class="confidence-label">Confidence</div>
                <div class="confidence-value">{conf:.1f}%</div>
                {render_bar(conf, bar)}
            </div>
            <div class="metric-row" style="justify-content:center;margin-top:1rem;">
                <div class="metric-chip">🔴 Fake probability<strong>{r['fake_pct']:.1f}%</strong></div>
                <div class="metric-chip">🟢 Real probability<strong>{r['real_pct']:.1f}%</strong></div>
                <div class="metric-chip metric-chip-bert">Model<strong>{"DistilRoBERTa API" if r["api_used"] else "TF-IDF+RF Local"}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


with tab_img:
    uploaded_img = st.file_uploader("UPLOAD IMAGE  (JPG / PNG / WEBP)", type=['jpg','jpeg','png','webp'], key="img_up")
    if uploaded_img and st.button("ANALYZE IMAGE →", key="btn_img"):
        with st.spinner("Running ELA forensics + AI-image API detection…"):
            try:
                raw_pil   = Image.open(uploaded_img)
                small_pil = resize_for_analysis(raw_pil, max_dim=900)
                ela       = perform_ela(small_pil)
                noise     = analyze_noise_consistency(small_pil)
                exif      = extract_exif_flags(raw_pil)
                classical = compute_ela_classical_score(ela, noise, exif)

                ai_prob, ai_err = None, None
                if key_ok:
                    ai_prob, ai_err = api_detect_ai_image(small_pil, hf_key)

                combined  = compute_image_forgery_score(classical, ai_prob)
                verdict, vcolor, vcircle = score_to_verdict(combined)
                ela_verd, ela_vc, ela_vc2 = score_to_verdict(classical)
                heatmap   = build_ela_figure(ela['ela_array'], small_pil)

                st.session_state['image_result'] = {
                    'combined': combined, 'classical': classical, 'ai_prob': ai_prob,
                    'ai_err': ai_err, 'verdict': verdict, 'vcolor': vcolor, 'vcircle': vcircle,
                    'ela_verd': ela_verd, 'ela_vc': ela_vc, 'ela_vc2': ela_vc2,
                    'ela': ela, 'noise': noise, 'exif': exif, 'heatmap': heatmap,
                }
            except Exception as exc:
                st.markdown(f'<div class="error-box">✕  Image analysis failed: {exc}</div>', unsafe_allow_html=True)

    ir = st.session_state['image_result']
    if ir:
        col_ela, col_ai = st.columns(2, gap="medium")

        with col_ela:
            ela_card = "result-fake" if ir['classical'] >= 65 else ("result-warning" if ir['classical'] >= 35 else "result-real")
            ela_bar  = "bar-fake"    if ir['classical'] >= 65 else ("bar-warn" if ir['classical'] >= 35 else "bar-real")
            st.markdown(f"""
            <div class="result-card {ela_card}">
                <div class="ela-badge">🔬 ELA Forensics  ·  Local (CPU)</div>
                <div class="verdict-emoji">{ir['ela_vc2']}</div>
                <div class="verdict-label">Manipulation Score</div>
                <div class="verdict-text" style="color:{ir['ela_vc']};">{ir['classical']}%</div>
                <div class="verdict-label" style="font-size:0.68rem;color:{ir['ela_vc']};margin-top:0.2rem;">{ir['ela_verd']}</div>
                <div class="confidence-section">
                    <div class="confidence-label">Detects: splicing · cloning · Photoshop · retouching</div>
                    {render_bar(ir['classical'], ela_bar)}
                </div>
                <div class="metric-row" style="justify-content:center;margin-top:0.8rem;flex-direction:column;gap:0.4rem;">
                    <div class="metric-chip">ELA Mean Error<strong>{ir['ela']['mean_error']:.3f}</strong></div>
                    <div class="metric-chip">Suspicious Pixels<strong>{ir['ela']['suspicious_pct']:.2f}%</strong></div>
                    <div class="metric-chip">EXIF Software<strong>{str(ir['exif']['software'])[:20]}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ai:
            if ir['ai_prob'] is not None:
                ap       = ir['ai_prob']
                av, avc, acircle = score_to_verdict(ap)
                ai_card  = "result-fake" if ap >= 65 else ("result-warning" if ap >= 35 else "result-real")
                ai_bar   = "bar-fake"    if ap >= 65 else ("bar-warn" if ap >= 35 else "bar-real")
                st.markdown(f"""
                <div class="result-card {ai_card}">
                    <div class="dl-badge">🤖 AI-Image Detector  ·  HF Inference API</div>
                    <div class="verdict-emoji">{acircle}</div>
                    <div class="verdict-label">AI Generation Score</div>
                    <div class="verdict-text" style="color:{avc};">{ap}%</div>
                    <div class="verdict-label" style="font-size:0.68rem;color:{avc};margin-top:0.2rem;">{av}</div>
                    <div class="confidence-section">
                        <div class="confidence-label">Detects: Midjourney · DALL·E · Stable Diffusion · Firefly · Flux</div>
                        {render_bar(ap, ai_bar)}
                    </div>
                    <div class="metric-row" style="justify-content:center;margin-top:0.8rem;flex-direction:column;gap:0.4rem;">
                        <div class="metric-chip metric-chip-ai">AI Generated<strong>{ap}%</strong></div>
                        <div class="metric-chip">Authentic / Real<strong>{round(100-ap,1)}%</strong></div>
                        <div class="metric-chip">Model<strong style="font-size:0.65rem;">prithivMLmods/AI-vs-Deepfake-vs-Real-9999</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if ir['exif']['has_editing_software']:
                    st.markdown('<div class="warn-box" style="margin-top:0.5rem;">⚠️  EXIF metadata confirms photo-editing software was used on this image.</div>', unsafe_allow_html=True)
            elif not key_ok:
                st.markdown('<div class="result-card result-neutral"><div class="verdict-emoji">🔑</div><div class="verdict-label">API Key Required</div><p style="color:#5a5a8a;font-size:0.84rem;">Set HF_API_KEY in Streamlit Secrets to enable the AI-image detector. See the Setup tab.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-card result-neutral"><div class="verdict-emoji">⚠️</div><div class="verdict-label">API Unavailable</div><p style="color:#5a5a8a;font-size:0.84rem;">{ir.get("ai_err","Model warming up — try again in 20 seconds.")}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">COMBINED SCORE  ·  70% DL Weight + 30% ELA Weight</div>', unsafe_allow_html=True)
        cv, cvc, ccircle = score_to_verdict(ir['combined'])
        comb_card = "result-fake" if ir['combined'] >= 65 else ("result-warning" if ir['combined'] >= 35 else "result-real")
        comb_bar  = "bar-fake"    if ir['combined'] >= 65 else ("bar-warn" if ir['combined'] >= 35 else "bar-real")
        st.markdown(f"""
        <div class="result-card {comb_card}">
            <div class="verdict-emoji">{ccircle}</div>
            <div class="verdict-label">Overall Image Risk</div>
            <div class="verdict-text" style="color:{cvc};font-size:2.8rem;">{ir['combined']}%</div>
            <div class="verdict-label" style="color:{cvc};">{cv}</div>
            <div class="confidence-section">{render_bar(ir['combined'], comb_bar)}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-header">ELA HEATMAP</div>', unsafe_allow_html=True)
        st.image(ir['heatmap'], use_container_width=True)


with tab_vid:
    st.markdown('<div class="warn-box">⚠️  Recommended: under 40 MB and under 2 minutes. API frame scoring samples 5 frames to stay within rate limits.</div>', unsafe_allow_html=True)
    uploaded_vid = st.file_uploader("UPLOAD VIDEO  (MP4 / AVI / MOV)", type=['mp4','avi','mov','mkv'], key="vid_up")
    if uploaded_vid and st.button("ANALYZE VIDEO →", key="btn_vid"):
        with st.spinner("Extracting frames · face detection · ELA · AI frame scoring via API…"):
            try:
                vid_bytes = uploaded_vid.read()
                frames, vmeta = extract_video_frames(vid_bytes, max_frames=12)
                if not frames:
                    st.markdown('<div class="error-box">✕  Could not extract frames. Check the video is not corrupted.</div>', unsafe_allow_html=True)
                else:
                    face_data    = [detect_faces(f) for f in frames]
                    ela_scores   = [frame_ela_score(f) for f in frames]
                    temp_score, diffs = temporal_inconsistency(frames)
                    total_faces  = sum(len(fd) for fd in face_data)
                    skin_score   = skin_tone_consistency(frames, face_data) if total_faces > 0 else 0.4
                    ai_vid_prob, ai_vid_err = None, None
                    if key_ok:
                        ai_vid_prob, ai_vid_err = analyze_video_frames_api(frames, hf_key, max_ai_frames=5)
                    combined, classical = compute_deepfake_score(ela_scores, temp_score, skin_score, total_faces, ai_vid_prob)
                    verdict, vcolor, vcircle = score_to_verdict(combined)
                    frame_fig = build_frame_figure(frames, face_data, ela_scores)
                    temp_fig  = build_temporal_figure(diffs) if len(diffs) > 1 else None
                    st.session_state['video_result'] = {
                        'combined': combined, 'classical': classical, 'ai_prob': ai_vid_prob,
                        'ai_err': ai_vid_err, 'verdict': verdict, 'vcolor': vcolor, 'vcircle': vcircle,
                        'vmeta': vmeta, 'n_frames': len(frames), 'total_faces': total_faces,
                        'temp_score': temp_score, 'skin_score': skin_score,
                        'ela_mean': float(np.mean([s[0] for s in ela_scores])),
                        'frame_fig': frame_fig, 'temp_fig': temp_fig,
                    }
            except Exception as exc:
                st.markdown(f'<div class="error-box">✕  Video analysis failed: {exc}</div>', unsafe_allow_html=True)

    vr = st.session_state['video_result']
    if vr:
        comb_p   = vr['combined']
        card_cls = "result-fake" if comb_p >= 65 else ("result-warning" if comb_p >= 35 else "result-real")
        bar_cls  = "bar-fake"    if comb_p >= 65 else ("bar-warn" if comb_p >= 35 else "bar-real")
        ai_chip  = f'<div class="metric-chip metric-chip-ai">🤖 AI Gen Prob (avg)<strong>{vr["ai_prob"]}%</strong></div>' if vr['ai_prob'] is not None else ('<div class="metric-chip">🤖 AI API<strong style="color:#5a5a8a;">Key needed</strong></div>' if not key_ok else f'<div class="metric-chip">🤖 AI API<strong style="color:#ef4444;">Error</strong></div>')
        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="verdict-emoji">{vr['vcircle']}</div>
            <div class="verdict-label">Deepfake / AI-Video Risk</div>
            <div class="verdict-text" style="color:{vr['vcolor']};">{vr['verdict']}</div>
            <div class="confidence-section">
                <div class="confidence-label">Combined Score  ·  70% DL + 30% Classical  ·  Classical alone: {vr['classical']:.1f}%</div>
                <div class="confidence-value">{comb_p}%</div>
                {render_bar(comb_p, bar_cls)}
            </div>
            <div class="metric-row" style="justify-content:center;margin-top:1rem;flex-wrap:wrap;">
                <div class="metric-chip">Frames<strong>{vr['n_frames']}</strong></div>
                <div class="metric-chip">Faces<strong>{vr['total_faces']}</strong></div>
                <div class="metric-chip">Duration<strong>{vr['vmeta'].get('duration_s','?')}s</strong></div>
                <div class="metric-chip">Mean ELA<strong>{vr['ela_mean']:.3f}</strong></div>
                <div class="metric-chip">Temporal<strong>{vr['temp_score']:.3f}</strong></div>
                {ai_chip}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if vr['ai_prob'] is not None and vr['ai_prob'] >= 50:
            st.markdown(f'<div class="warn-box" style="border-color:rgba(239,68,68,0.3);color:#fca5a5;background:rgba(239,68,68,0.05);">🤖 <strong>DL Override active:</strong> Average AI generation probability across frames is <strong>{vr["ai_prob"]}%</strong> — elevated to HIGH RISK. Consistent with AI text-to-video generation (Sora, Veo, Runway).</div>', unsafe_allow_html=True)
        if vr['frame_fig']:
            st.markdown('<div class="section-header">EXTRACTED FRAMES</div>', unsafe_allow_html=True)
            st.image(vr['frame_fig'], use_container_width=True)
        if vr['temp_fig']:
            st.markdown('<div class="section-header">TEMPORAL CONSISTENCY</div>', unsafe_allow_html=True)
            st.image(vr['temp_fig'], use_container_width=True)


with tab_cross:
    st.markdown('<div class="section-header">CLIP-POWERED TEXT — IMAGE SEMANTIC MATCH</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box" style="margin-bottom:1.2rem;">
        <strong>Model:</strong> <code>openai/clip-vit-base-patch32</code> via HF Inference API.<br>
        CLIP computes cosine similarity between a text embedding and an image embedding in a shared 512-dimensional multimodal space.
        A low score means the image content does not semantically match the headline — a core disinformation tactic.
        {"<br><strong>Status:</strong> ✓ API key configured — CLIP analysis active." if key_ok else "<br><strong>Status:</strong> ✗ API key not set — CLIP analysis unavailable. Configure in Setup tab."}
    </div>
    """, unsafe_allow_html=True)

    cross_text = st.text_input("NEWS HEADLINE / CLAIM", placeholder="e.g.  'Massive flooding kills hundreds in coastal city'", key="cross_t")
    cross_img  = st.file_uploader("UPLOAD ACCOMPANYING IMAGE", type=['jpg','jpeg','png','webp'], key="cross_i")

    if cross_text and cross_img and st.button("CHECK SEMANTIC MATCH →", key="btn_cross"):
        if not key_ok:
            st.markdown('<div class="error-box">✕  HF_API_KEY required for CLIP analysis. Configure it in the Setup tab.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Calling CLIP API for cross-modal similarity…"):
                try:
                    pil   = Image.open(cross_img)
                    small = resize_for_analysis(pil, max_dim=600)
                    sim_pct, mismatch_pct, clip_err = api_clip_similarity(small, cross_text, hf_key)
                    if clip_err or sim_pct is None:
                        st.markdown(f'<div class="error-box">✕  CLIP API error: {clip_err}</div>', unsafe_allow_html=True)
                        st.session_state['cross_result'] = None
                    else:
                        mismatch_score = mismatch_pct
                        verdict, vcolor, vcircle = score_to_verdict(mismatch_score)
                        st.session_state['cross_result'] = {
                            'sim_pct': sim_pct, 'mismatch_score': mismatch_score,
                            'verdict': verdict, 'vcolor': vcolor, 'vcircle': vcircle,
                            'headline': cross_text[:80],
                        }
                except Exception as exc:
                    st.markdown(f'<div class="error-box">✕  Analysis failed: {exc}</div>', unsafe_allow_html=True)

    cr = st.session_state['cross_result']
    if cr:
        card_cls = "result-fake" if cr['mismatch_score'] >= 65 else ("result-warning" if cr['mismatch_score'] >= 35 else "result-real")
        bar_cls  = "bar-fake"    if cr['mismatch_score'] >= 65 else ("bar-warn" if cr['mismatch_score'] >= 35 else "bar-real")
        bar_sim  = "bar-real"    if cr['sim_pct'] >= 65 else ("bar-warn" if cr['sim_pct'] >= 35 else "bar-fake")

        st.markdown(f"""
        <div class="result-card {card_cls}" style="margin-top:1.2rem;">
            <div class="dl-badge">🎬 CLIP · openai/clip-vit-base-patch32 · HF Inference API</div>
            <div class="verdict-emoji">{cr['vcircle']}</div>
            <div class="verdict-label">Text–Image Mismatch Risk</div>
            <div class="verdict-text" style="color:{cr['vcolor']};">{cr['verdict']}</div>
        </div>
        """, unsafe_allow_html=True)

        c_sim, c_mis = st.columns(2, gap="medium")
        with c_sim:
            st.markdown(f"""
            <div class="similarity-meter">
                <div class="similarity-label">CLIP Semantic Match</div>
                <div class="similarity-value" style="color:#34d399;">{cr['sim_pct']}%</div>
                <div style="font-size:0.75rem;color:#5a5a7a;margin:0.4rem 0;">how well the image represents the headline</div>
                {render_bar(cr['sim_pct'], bar_sim)}
            </div>
            """, unsafe_allow_html=True)
        with c_mis:
            st.markdown(f"""
            <div class="similarity-meter">
                <div class="similarity-label">Mismatch Risk Score</div>
                <div class="similarity-value" style="color:{cr['vcolor']};">{cr['mismatch_score']}%</div>
                <div style="font-size:0.75rem;color:#5a5a7a;margin:0.4rem 0;">used in the overall multimodal risk summary</div>
                {render_bar(cr['mismatch_score'], bar_cls)}
            </div>
            """, unsafe_allow_html=True)

        if cr['mismatch_score'] >= 65:
            st.markdown(f"""
            <div class="warn-box" style="border-color:rgba(239,68,68,0.3);color:#fca5a5;background:rgba(239,68,68,0.05);">
                🚨  <strong>HIGH MISMATCH DETECTED:</strong> CLIP gives only <strong>{cr['sim_pct']}%</strong> semantic similarity 
                between the image and the headline "<em>{cr['headline']}…</em>". This pattern — attaching an unrelated 
                image to a fabricated headline — is a primary disinformation technique.
            </div>
            """, unsafe_allow_html=True)
        elif cr['mismatch_score'] >= 35:
            st.markdown(f'<div class="warn-box">⚠️  CLIP similarity is moderate ({cr["sim_pct"]}%). The image may be loosely related but does not strongly represent the specific claim in the headline.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">✓  CLIP confirms high semantic alignment ({cr["sim_pct"]}%) — the image content is consistent with the headline.</div>', unsafe_allow_html=True)


with tab_setup:
    st.markdown('<div class="section-header">STEP-BY-STEP API KEY SETUP</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="margin-bottom:1.5rem;">
        <strong>All DL inference in this app is offloaded to the Hugging Face Inference API.</strong><br>
        Your Streamlit Cloud server uses zero RAM for model weights. The free HF Inference API tier 
        includes generous monthly compute — sufficient for a college project demo.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ##### Step 1 — Get a Free Hugging Face API Key
    1. Go to **[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**
    2. Click **"New token"**
    3. Name it `streamlit-fake-news`, set Role = **Read**
    4. Click **Generate** and copy the token (starts with `hf_…`)
    """)

    st.markdown("""
    ##### Step 2 — Add it to Streamlit Cloud Secrets
    In Streamlit Community Cloud (`share.streamlit.io`):
    1. Open your deployed app → click the **⋮ menu** (three dots top-right) → **Settings**
    2. Click **Secrets** in the left panel
    3. Paste exactly this, replacing `hf_YOUR_TOKEN_HERE`:
    """)

    st.code('[secrets]\nHF_API_KEY = "hf_YOUR_TOKEN_HERE"', language="toml")

    st.markdown("""
    ##### Step 3 — For Local Development
    Create a file at `.streamlit/secrets.toml` in your project root:
    """)
    st.code('HF_API_KEY = "hf_YOUR_TOKEN_HERE"', language="toml")

    st.markdown("""
    ##### Step 4 — Verify it Works
    After saving secrets, reboot the app from Streamlit Cloud dashboard. The header should show
    **"✓ HF_API_KEY configured — API models active"**.

    ##### API Models Used & What They Do
    """)

    st.markdown("""
    | Module | Model | Task | Why It Was Chosen |
    |---|---|---|---|
    | 📝 Text | `vikram71198/distilroberta-base-finetuned-fake-news-detection` | Text classification | Fine-tuned on 24K+ labeled real/fake news articles; ~90%+ accuracy |
    | 🖼️ Image | `prithivMLmods/AI-vs-Deepfake-vs-Real-9999` | Image classification | ViT fine-tuned on real vs AI-generated image pairs (Midjourney, DALL·E, SD) |
    | 🎥 Video | Same as image — applied per frame | Image classification | Frame-level AI generation scoring; averages across 5 sampled frames |
    | 🔗 Cross-Modal | `openai/clip-vit-base-patch32` | Zero-shot image classification | Shared text+image embedding space; true cosine similarity between headline and image |

    ##### Handling API 503 "Model Loading" Errors
    HF free-tier models go to sleep after inactivity. The app automatically:
    - Retries up to **3 times** with exponential back-off (8s → 16s → 24s)
    - Falls back to the **local TF-IDF + Random Forest model** if the text API fails
    - Shows a yellow warning card if image/video API fails, displaying the ELA-only result
    - Advises you to **wait 20–30 seconds** and click Analyze again — the model will be warm
    """)

    st.markdown('<div class="section-header">RAM BUDGET ON STREAMLIT CLOUD (1 GB LIMIT)</div>', unsafe_allow_html=True)
    st.markdown("""
    | Component | RAM Used |
    |---|---|
    | Streamlit runtime + Python | ~120 MB |
    | OpenCV (headless) | ~45 MB |
    | scikit-learn local fallback model | ~30 MB |
    | PIL / Matplotlib | ~25 MB |
    | HF API (text, image, CLIP) | **0 MB** — runs on HF servers |
    | **Total** | **~220 MB / 1024 MB** ✅ |
    """)


any_result = any(st.session_state[k] for k in ['text_result','image_result','video_result','cross_result'])
if any_result:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">MULTIMODAL RISK SUMMARY</div>', unsafe_allow_html=True)

    tr = st.session_state['text_result']
    ir = st.session_state['image_result']
    vr = st.session_state['video_result']
    cr = st.session_state['cross_result']

    ts = tr['fake_pct'] if tr else None
    is_ = ir['combined'] if ir else None
    vs  = vr['combined'] if vr else None
    cs  = cr['mismatch_score'] if cr else None

    active  = [s for s in [ts, is_, vs, cs] if s is not None]
    overall = round(sum(active) / len(active), 1) if active else 0
    ov, ovc, ocircle = score_to_verdict(overall)
    ov_bar = "bar-fake" if overall >= 65 else ("bar-warn" if overall >= 35 else "bar-real")

    def modal_tile(label, score, note=""):
        if score is None:
            return f'<div class="modal-tile"><div class="modal-tile-label">{label}</div><div class="modal-tile-score" style="color:#2a2a4a;font-size:1rem;">N/A</div></div>'
        v, vc, _ = score_to_verdict(score)
        return f'<div class="modal-tile"><div class="modal-tile-label">{label}</div><div class="modal-tile-score" style="color:{vc};">{score}%</div><div class="modal-tile-verdict">{v}{note}</div></div>'

    ai_img_note = f" · DL:{ir['ai_prob']}%" if ir and ir.get('ai_prob') is not None else ""
    ai_vid_note = f" · DL:{vr['ai_prob']}%" if vr and vr.get('ai_prob') is not None else ""
    sim_note    = f" · sim:{cr['sim_pct']}%" if cr else ""

    st.markdown(f"""
    <div class="combined-card">
        <div class="combined-title">🛡️ Overall Multimodal Risk Assessment</div>
        <div class="modal-row">
            {modal_tile("TEXT ANALYSIS", ts)}
            {modal_tile("IMAGE FORENSICS", is_, ai_img_note)}
            {modal_tile("VIDEO DEEPFAKE", vs, ai_vid_note)}
            {modal_tile("CLIP MATCH", cs, sim_note)}
        </div>
        <div style="text-align:center;margin-top:1.2rem;">
            <div class="verdict-label">COMBINED RISK SCORE</div>
            <div class="verdict-text" style="color:{ovc};font-size:3rem;">{overall}%</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:{ovc};margin-bottom:0.8rem;">{ocircle} {ov}</div>
            <div class="progress-bar-bg" style="max-width:420px;margin:0 auto;">
                <div class="{ov_bar}" style="width:{int(overall)}%;height:8px;border-radius:100px;"></div>
            </div>
            <div style="font-size:0.72rem;color:#2e2e5a;margin-top:0.6rem;">
                Mean across {len(active)} active module{"s" if len(active)!=1 else ""}
                &nbsp;·&nbsp; Transformer API + CLIP + ELA + OpenCV
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr class="divider">
<p style="text-align:center;color:#1a1a3a;font-size:0.72rem;letter-spacing:1px;">
MULTIMODAL FAKE NEWS &amp; DEEPFAKE DETECTION v3 &nbsp;·&nbsp;
DistilRoBERTa · AI-Image-Detector · CLIP · ELA · Haar Cascade · Random Forest · Isolation Forest
</p>
""", unsafe_allow_html=True)
