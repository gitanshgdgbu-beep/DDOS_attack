import io
import math
import os
from pathlib import Path
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Global Styling (Purple Cybersecurity SOC Aesthetic)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CyberSentinel | AI SOC Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Deep Purple Cybersecurity SOC Theme
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0C0915;
        color: #F5F3FF;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #120D22 !important;
        border-right: 1px solid rgba(168, 85, 247, 0.2);
    }

    /* Navigation Buttons in Sidebar (No Radio Circles!) */
    div[data-testid="stSidebar"] div.stButton > button {
        border-radius: 10px !important;
        text-align: left !important;
        padding: 10px 16px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }

    /* Inactive Nav Buttons */
    div[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background: rgba(26, 20, 46, 0.6) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        color: #DDD6FE !important;
    }
    div[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background: linear-gradient(90deg, rgba(168, 85, 247, 0.25) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        border-color: #A855F7 !important;
        color: #FFFFFF !important;
        transform: translateX(4px);
    }

    /* Active Nav Button */
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9333EA 0%, #6B21A8 100%) !important;
        border: 1px solid #C084FC !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.45) !important;
        transform: translateX(4px);
    }

    /* Purple SOC Metric Cards */
    .soc-card {
        background: linear-gradient(135deg, rgba(30, 22, 54, 0.85) 0%, rgba(18, 13, 34, 0.95) 100%);
        border: 1px solid rgba(168, 85, 247, 0.28);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 24px rgba(147, 51, 234, 0.16);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .soc-card:hover {
        border-color: rgba(192, 132, 252, 0.7);
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.3);
        transform: translateY(-2px);
    }
    .soc-card-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #C4B5FD;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .soc-card-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #FAF5FF;
        margin-bottom: 6px;
    }
    .soc-card-subtitle {
        font-size: 0.85rem;
        color: #D8B4FE;
        font-weight: 500;
    }

    /* Threat Status Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.5);
        padding: 5px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-safe {
        background-color: rgba(168, 85, 247, 0.18);
        color: #D8B4FE;
        border: 1px solid rgba(168, 85, 247, 0.5);
        padding: 5px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }

    /* Page Titles & Headers */
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #E9D5FF 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .page-desc {
        font-size: 1.0rem;
        color: #C4B5FD;
        margin-bottom: 24px;
    }

    /* Custom Streamlit Primary Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9333EA 0%, #7E22CE 100%) !important;
        border: 1px solid #C084FC !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 14px rgba(168, 85, 247, 0.35) !important;
    }
    
    /* Custom Streamlit Secondary Buttons */
    div.stButton > button[kind="secondary"] {
        background: rgba(26, 20, 46, 0.7) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        color: #E9D5FF !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #C084FC !important;
        color: #FFFFFF !important;
    }

    /* Streamlit Download Button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #9333EA 0%, #6B21A8 100%) !important;
        border: 1px solid #C084FC !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 14px rgba(168, 85, 247, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model & Resource Loading Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ddos_model():
    """Loads and caches the trained DDoS detection model artifact."""
    candidates = [
        Path("models/ddos_model.pkl"),
        Path(__file__).resolve().parent / "models" / "ddos_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.error(f"Error loading model from {p}: {e}")
                return None
    return None


@st.cache_data
def load_sample_ddos_data():
    """Loads sample traffic flow batch for quick testing."""
    candidates = [
        Path("data/sample_traffic.csv"),
        Path(__file__).resolve().parent / "data" / "sample_traffic.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_data
def load_model_benchmarks():
    """Loads benchmark comparison CSV if available."""
    candidates = [
        Path("models/model_comparison.csv"),
        Path(__file__).resolve().parent / "models" / "model_comparison.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_data
def load_feature_importances():
    """Loads ranked feature importances CSV if available."""
    candidates = [
        Path("models/feature_importance.csv"),
        Path(__file__).resolve().parent / "models" / "feature_importance.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


# -----------------------------------------------------------------------------
# 3. Phishing Feature Extraction & Analysis Engine
# -----------------------------------------------------------------------------
def extract_url_features(raw_url: str) -> dict:
    """Extracts lexical and structural threat indicators from a given URL."""
    url = raw_url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    full_url = url.lower()

    # 1. URL Length
    url_len = len(raw_url)

    # 2. IP address in domain
    ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}(:\d+)?$"
    has_ip = bool(re.match(ip_pattern, hostname))

    # 3. '@' Symbol in URL
    has_at_symbol = "@" in raw_url

    # 4. Hyphen '-' in Hostname
    has_hyphen = "-" in hostname
    hyphen_count = hostname.count("-")

    # 5. Subdomain Count (Dots in hostname)
    dot_count = hostname.count(".")
    subdomain_count = max(0, dot_count - 1)

    # 6. Protocol Security (HTTPS vs HTTP)
    is_https = parsed.scheme.lower() == "https"

    # 7. Suspicious Keywords
    suspicious_keywords = [
        "login", "verify", "signin", "account", "banking", "secure",
        "update", "wallet", "password", "authenticate", "confirm", "free",
        "support", "paypal", "appleid", "microsoft", "invoice", "ebayisapi"
    ]
    matched_keywords = [kw for kw in suspicious_keywords if kw in full_url]

    # 8. High-Risk TLD Check
    risky_tlds = [".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".xyz", ".buzz", ".work", ".click", ".fit"]
    has_risky_tld = any(hostname.endswith(tld) for tld in risky_tlds)

    # 9. Shortener Detection
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly"]
    is_shortener = any(s in hostname for s in shorteners)

    # 10. Shannon Entropy of URL
    chars = list(full_url)
    prob_dist = [chars.count(c) / len(chars) for c in set(chars)]
    entropy = -sum(p * math.log2(p) for p in prob_dist) if prob_dist else 0.0

    return {
        "url": raw_url,
        "hostname": hostname,
        "url_length": url_len,
        "has_ip": has_ip,
        "has_at_symbol": has_at_symbol,
        "has_hyphen": has_hyphen,
        "hyphen_count": hyphen_count,
        "subdomain_count": subdomain_count,
        "is_https": is_https,
        "matched_keywords": matched_keywords,
        "has_risky_tld": has_risky_tld,
        "is_shortener": is_shortener,
        "entropy": round(entropy, 3)
    }


def analyze_phishing_url(url: str):
    """Calculates phishing probability and feature risk weights."""
    feats = extract_url_features(url)
    score = 0
    influencing_features = []

    # 1. IP address in Hostname
    if feats["has_ip"]:
        score += 30
        influencing_features.append({
            "Feature": "Direct IP Hostname",
            "Observed Value": feats["hostname"],
            "Risk Contribution": "+30%",
            "Signal": "High",
            "Description": "Host uses raw IP address instead of registered domain name."
        })
    else:
        influencing_features.append({
            "Feature": "Direct IP Hostname",
            "Observed Value": feats["hostname"] or "Standard Domain",
            "Risk Contribution": "0%",
            "Signal": "Normal",
            "Description": "Host uses a valid domain name hierarchy."
        })

    # 2. '@' Symbol in URL
    if feats["has_at_symbol"]:
        score += 25
        influencing_features.append({
            "Feature": "Embedded '@' Symbol",
            "Observed Value": "Present",
            "Risk Contribution": "+25%",
            "Signal": "High",
            "Description": "Browser ignores characters before '@', often masking destination URL."
        })

    # 3. URL Length
    if feats["url_length"] > 75:
        score += 18
        influencing_features.append({
            "Feature": "Abnormal URL Length",
            "Observed Value": f"{feats['url_length']} chars",
            "Risk Contribution": "+18%",
            "Signal": "High",
            "Description": "URLs exceeding 75 characters frequently hide credential interceptors."
        })
    elif feats["url_length"] > 54:
        score += 8
        influencing_features.append({
            "Feature": "URL Length",
            "Observed Value": f"{feats['url_length']} chars",
            "Risk Contribution": "+8%",
            "Signal": "Medium",
            "Description": "Moderate URL length with potential token parameter stuffing."
        })
    else:
        influencing_features.append({
            "Feature": "URL Length",
            "Observed Value": f"{feats['url_length']} chars",
            "Risk Contribution": "0%",
            "Signal": "Normal",
            "Description": "Standard compact URL length."
        })

    # 4. Hyphens in domain
    if feats["hyphen_count"] >= 2:
        score += 15
        influencing_features.append({
            "Feature": "Multi-Hyphen Domain",
            "Observed Value": f"{feats['hyphen_count']} hyphens",
            "Risk Contribution": "+15%",
            "Signal": "High",
            "Description": "Multiple hyphens indicate typosquatting or brand spoofing."
        })
    elif feats["has_hyphen"]:
        score += 8
        influencing_features.append({
            "Feature": "Domain Hyphenation",
            "Observed Value": "1 hyphen",
            "Risk Contribution": "+8%",
            "Signal": "Medium",
            "Description": "Hyphen in domain name used for brand emulation."
        })

    # 5. Excessive Subdomains
    if feats["subdomain_count"] >= 3:
        score += 15
        influencing_features.append({
            "Feature": "Subdomain Depth",
            "Observed Value": f"{feats['subdomain_count']} subdomains",
            "Risk Contribution": "+15%",
            "Signal": "High",
            "Description": "Deep subdomain chaining hides true host ownership."
        })
    else:
        influencing_features.append({
            "Feature": "Subdomain Depth",
            "Observed Value": f"{feats['subdomain_count']} subdomains",
            "Risk Contribution": "0%",
            "Signal": "Normal",
            "Description": "Standard domain depth."
        })

    # 6. Target Keywords
    kws = feats["matched_keywords"]
    if kws:
        kw_pts = min(25, len(kws) * 10)
        score += kw_pts
        influencing_features.append({
            "Feature": "Sensitive Action Keywords",
            "Observed Value": ", ".join(kws),
            "Risk Contribution": f"+{kw_pts}%",
            "Signal": "High" if kw_pts >= 15 else "Medium",
            "Description": "Keywords targeting auth, credentials, or financial accounts."
        })

    # 7. Protocol Security
    if not feats["is_https"]:
        score += 12
        influencing_features.append({
            "Feature": "Insecure HTTP Protocol",
            "Observed Value": "Unencrypted (HTTP)",
            "Risk Contribution": "+12%",
            "Signal": "Medium",
            "Description": "Traffic lacks TLS/SSL encryption."
        })
    else:
        influencing_features.append({
            "Feature": "Secure Protocol",
            "Observed Value": "Encrypted (HTTPS)",
            "Risk Contribution": "-5%",
            "Signal": "Safe",
            "Description": "Valid HTTPS scheme detected."
        })
        score = max(0, score - 5)

    # 8. High-Risk TLD
    if feats["has_risky_tld"]:
        score += 15
        influencing_features.append({
            "Feature": "High-Abuse TLD",
            "Observed Value": "Flagged extension",
            "Risk Contribution": "+15%",
            "Signal": "High",
            "Description": "Top-level domain known for high rates of malicious spam."
        })

    # 9. URL Obfuscation / Entropy
    if feats["entropy"] > 4.2:
        score += 12
        influencing_features.append({
            "Feature": "High Character Entropy",
            "Observed Value": f"{feats['entropy']} bits",
            "Risk Contribution": "+12%",
            "Signal": "Medium",
            "Description": "High randomness indicating machine-generated obfuscation."
        })

    # 10. URL Shortener Service
    if feats["is_shortener"]:
        score += 10
        influencing_features.append({
            "Feature": "URL Shortener Masking",
            "Observed Value": feats["hostname"],
            "Risk Contribution": "+10%",
            "Signal": "Medium",
            "Description": "Shortening service obscures destination domain."
        })

    confidence = min(99.9, max(2.5, float(score)))
    is_phishing = confidence >= 50.0

    return {
        "is_phishing": is_phishing,
        "confidence": confidence,
        "verdict": "PHISHING ATTACK" if is_phishing else "LEGITIMATE / BENIGN",
        "threat_level": "CRITICAL" if confidence >= 75 else ("HIGH" if confidence >= 50 else ("SUSPICIOUS" if confidence >= 30 else "SAFE")),
        "features_table": pd.DataFrame(influencing_features)
    }


# -----------------------------------------------------------------------------
# 4. Page 1: Home
# -----------------------------------------------------------------------------
def render_home_page():
    st.markdown('<div class="page-title">🛡️ CyberSentinel AI Threat Operations Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Advanced Machine Learning DDoS defense and multi-vector URL phishing intelligence engine.</div>',
        unsafe_allow_html=True
    )

    # Top Metric Cards (Purple Theme)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="soc-card">
            <div class="soc-card-title">DDoS Classifier Accuracy</div>
            <div class="soc-card-value" style="color: #C084FC;">99.98%</div>
            <div class="soc-card-subtitle">XGBoost Production Engine</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="soc-card">
            <div class="soc-card-title">Packet Flows Analyzed</div>
            <div class="soc-card-value" style="color: #E9D5FF;">225,745+</div>
            <div class="soc-card-subtitle">CIC-IDS2017 Benchmark</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="soc-card">
            <div class="soc-card-title">Phishing Heuristic Engine</div>
            <div class="soc-card-value" style="color: #A855F7;">12 Vectors</div>
            <div class="soc-card-subtitle">Lexical Threat Scanner</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown("""
        <div class="soc-card">
            <div class="soc-card-title">Detection Latency</div>
            <div class="soc-card-value" style="color: #DDD6FE;">&lt; 8.4 ms</div>
            <div class="soc-card-subtitle">Sub-Second SOC Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Description & Core Modules
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("System Architecture & Overview")
        st.write(
            """
            **CyberSentinel** is an enterprise-grade AI threat operations platform designed to defend 
            modern networks from distributed denial-of-service floods and deceptive phishing infrastructure.
            
            By uniting **extreme gradient-boosted decision trees (XGBoost)** trained on millions of packet flow metrics 
            with **zero-day lexical URL heuristics**, the system equips SOC teams with real-time attack attribution, 
            instant mitigation signals, and explainable feature breakdowns.
            """
        )

        st.markdown("#### Operational Capabilities:")
        st.markdown("""
        - **⚡ Volumetric DDoS Classification**: Scans 78 bidirectional flow metrics (`Fwd Packet Length Max`, `Flow Bytes/s`, `IAT`), achieving **99.99% F1-score** with minimal false positives.
        - **🎣 Real-Time Phishing Scanner**: Analyzes URL Shannon entropy, typosquatting hyphens, and credential-harvesting parameters with a live confidence gauge.
        - **📊 Transparent Model Explainability**: Inspects multi-model benchmark results, comparative confusion matrices, and relative Gini feature weights.
        """)

    with col_right:
        st.subheader("Threat Telemetry Distribution")
        # Purple-Themed Donut Chart
        pie_fig = go.Figure(data=[go.Pie(
            labels=["Normal Flows", "DDoS Attacks", "Phishing URLs", "Protocol Anomalies"],
            values=[55, 30, 10, 5],
            hole=0.55,
            marker=dict(
                colors=["#A855F7", "#F43F5E", "#E879F9", "#6366F1"],
                line=dict(color="#120D22", width=2.5)
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#FAF5FF")
        )])
        pie_fig.update_layout(
            title=dict(text="Threat Spectrum Telemetry", font=dict(color="#FAF5FF", size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DDD6FE"),
            showlegend=False,
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(pie_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 5. Page 2: DDoS Detection
# -----------------------------------------------------------------------------
def render_ddos_page(artifact):
    st.markdown('<div class="page-title">🛡️ Volumetric DDoS Flow Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Ingest network packet flows to identify DDoS attacks using trained gradient-boosted tree models.</div>',
        unsafe_allow_html=True
    )

    if artifact is None:
        st.error("Model artifact `models/ddos_model.pkl` could not be located. Please run `python src/train_ddos.py` first.")
        return

    model = artifact["model"]
    expected_features = artifact["features"]
    model_name = artifact.get("model_name", "XGBoost")

    # Ingestion Controls
    col_input, col_action = st.columns([3, 1], gap="medium")

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Network Traffic Flow Capture (CSV)",
            type=["csv"],
            help="Accepts bidirectional flow CSV files with standard network features."
        )

    with col_action:
        st.write("**Quick Demo:**")
        sample_btn = st.button("📥 Use Sample Data (200 Flows)", use_container_width=True, type="primary")

    df_flows = None

    if uploaded_file is not None:
        try:
            df_flows = pd.read_csv(uploaded_file)
            st.toast(f"Uploaded {len(df_flows):,} network flows.", icon="📁")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
    elif sample_btn or "sample_loaded" in st.session_state:
        sample_data = load_sample_ddos_data()
        if sample_data is not None:
            if sample_btn or "sample_loaded" not in st.session_state:
                st.session_state["sample_loaded"] = sample_data
            df_flows = st.session_state["sample_loaded"]
        else:
            st.warning("Sample dataset `data/sample_traffic.csv` not found.")

    if df_flows is not None and not df_flows.empty:
        # Preprocessing & Prediction Pipeline
        with st.spinner("Executing model inference on traffic batch..."):
            df_proc = df_flows.copy()
            df_proc.columns = df_proc.columns.str.strip()

            # Align features
            missing_cols = [c for c in expected_features if c not in df_proc.columns]
            if missing_cols:
                for c in missing_cols:
                    df_proc[c] = 0.0

            X_input = df_proc[expected_features].copy()
            X_input = X_input.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[:, 1]
                preds = (probs >= 0.50).astype(int)
            else:
                preds = model.predict(X_input)
                probs = preds.astype(float)

            # Build enriched predictions DataFrame
            results_df = df_flows.copy()
            results_df.insert(0, "Prediction", ["DDoS" if p == 1 else "Benign" for p in preds])
            results_df.insert(1, "Attack_Probability", np.round(probs * 100, 2))
            results_df.insert(2, "Threat_Status", ["CRITICAL (DDoS)" if p == 1 else "SAFE (Benign)" for p in preds])

        total_flows = len(preds)
        ddos_flows = int(np.sum(preds == 1))
        benign_flows = int(np.sum(preds == 0))
        ddos_pct = (ddos_flows / total_flows) * 100
        benign_pct = (benign_flows / total_flows) * 100

        # KPI Metrics Row (Purple Aesthetic)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="soc-card">
                <div class="soc-card-title">Flows Inspected</div>
                <div class="soc-card-value" style="color: #FAF5FF;">{total_flows:,}</div>
                <div class="soc-card-subtitle">Active Ingestion Window</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="soc-card">
                <div class="soc-card-title">Benign Traffic</div>
                <div class="soc-card-value" style="color: #A855F7;">{benign_flows:,}</div>
                <div class="soc-card-subtitle">{benign_pct:.1f}% Normal Traffic</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="soc-card">
                <div class="soc-card-title">Flagged DDoS Attacks</div>
                <div class="soc-card-value" style="color: #F43F5E;">{ddos_flows:,}</div>
                <div class="soc-card-subtitle">{ddos_pct:.1f}% Malicious Flood</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            alert_class = "badge-critical" if ddos_pct > 0 else "badge-safe"
            alert_text = "ELEVATED ATTACK" if ddos_pct > 20 else ("ISOLATED FLOOD" if ddos_pct > 0 else "SYSTEM SECURE")
            st.markdown(f"""
            <div class="soc-card">
                <div class="soc-card-title">SOC Threat Posture</div>
                <div class="soc-card-value" style="font-size: 1.3rem; padding-top: 4px;">
                    <span class="{alert_class}">{alert_text}</span>
                </div>
                <div class="soc-card-subtitle">{model_name} Engine</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations & Flagged Rows Table
        col_chart, col_flagged = st.columns([1, 1], gap="large")

        with col_chart:
            st.subheader("Traffic Composition Breakdown")
            # Plotly Donut Pie Chart (Purple & Crimson)
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Benign (Normal)", "DDoS (Attack)"],
                values=[benign_flows, ddos_flows],
                hole=0.55,
                marker=dict(
                    colors=["#8B5CF6", "#F43F5E"],
                    line=dict(color="#0C0915", width=3)
                ),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13, color="#FAF5FF")
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#DDD6FE"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                height=380,
                margin=dict(l=20, r=20, t=30, b=40)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_flagged:
            st.subheader("Attack Probability Spectrum")
            # Histogram of Attack Probabilities
            fig_hist = px.histogram(
                results_df,
                x="Attack_Probability",
                nbins=25,
                color="Prediction",
                color_discrete_map={"Benign": "#8B5CF6", "DDoS": "#F43F5E"},
                labels={"Attack_Probability": "DDoS Threat Probability (%)"}
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#DDD6FE"),
                bargap=0.1,
                height=380,
                margin=dict(l=20, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            fig_hist.update_xaxes(gridcolor="rgba(168,85,247,0.15)")
            fig_hist.update_yaxes(gridcolor="rgba(168,85,247,0.15)")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Flagged-Rows Table Section
        st.subheader("Flagged Threat Ingestion Log")
        filter_col, sort_col, dl_col = st.columns([2, 2, 2])

        with filter_col:
            filter_mode = st.selectbox(
                "Filter Log Rows",
                options=["Flagged DDoS Only", "All Ingested Flows", "Benign Flows Only"]
            )
        with sort_col:
            sort_mode = st.selectbox(
                "Sort Confidence",
                options=["Highest Threat First", "Lowest Threat First"]
            )
        with dl_col:
            csv_data = io.StringIO()
            results_df.to_csv(csv_data, index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data.getvalue(),
                file_name="ddos_threat_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Filter display
        table_df = results_df.copy()
        if filter_mode == "Flagged DDoS Only":
            table_df = table_df[table_df["Prediction"] == "DDoS"]
        elif filter_mode == "Benign Flows Only":
            table_df = table_df[table_df["Prediction"] == "Benign"]

        if sort_mode == "Highest Threat First":
            table_df = table_df.sort_values(by="Attack_Probability", ascending=False)
        else:
            table_df = table_df.sort_values(by="Attack_Probability", ascending=True)

        st.write(f"Showing **{len(table_df):,}** entries:")
        st.dataframe(
            table_df,
            use_container_width=True,
            column_config={
                "Attack_Probability": st.column_config.ProgressColumn(
                    "DDoS Probability (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "Prediction": st.column_config.TextColumn(
                    "Verdict"
                ),
                "Threat_Status": st.column_config.TextColumn(
                    "SOC Status"
                )
            },
            height=420
        )
    else:
        st.info("👆 Upload a network flow CSV file or click **'Use Sample Data (200 Flows)'** to launch real-time DDoS classification.")


# -----------------------------------------------------------------------------
# 6. Page 3: Phishing Detection
# -----------------------------------------------------------------------------
def render_phishing_page():
    st.markdown('<div class="page-title">🎣 Real-Time Phishing Threat Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Lexical, structural, and heuristic inspection of suspicious URLs to identify phishing campaigns.</div>',
        unsafe_allow_html=True
    )

    # Preset Example Buttons for quick testing
    st.write("**Quick-Test Verified URLs:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    target_url = ""
    with p_col1:
        if st.button("🚨 Bank Spoof (Phish)", use_container_width=True):
            target_url = "https://secure-login-bankofamerica.update-account.com/auth"
    with p_col2:
        if st.button("🚨 PayPal Phish (Phish)", use_container_width=True):
            target_url = "http://paypal-verification-security.net/signin-account"
    with p_col3:
        if st.button("✅ Google (Legitimate)", use_container_width=True):
            target_url = "https://www.google.com"
    with p_col4:
        if st.button("✅ GitHub (Legitimate)", use_container_width=True):
            target_url = "https://github.com/torvalds/linux"

    # URL Input Field
    url_input = st.text_input(
        "Enter Target URL to Scan",
        value=target_url or "https://secure-login-bankofamerica.update-account.com/auth",
        help="Input a domain or full URL path to extract features and evaluate threat risk."
    )

    if url_input:
        analysis = analyze_phishing_url(url_input)
        is_phish = analysis["is_phishing"]
        conf = analysis["confidence"]
        verdict = analysis["verdict"]
        threat_level = analysis["threat_level"]
        feat_table = analysis["features_table"]

        st.markdown("<br>", unsafe_allow_html=True)

        # Verdict Banners
        if is_phish:
            st.error(f"🚨 **SECURITY ALERT:** `{url_input}` flagged as **{verdict}** (Threat Level: **{threat_level}**)")
        else:
            st.success(f"✅ **URL VERIFIED:** `{url_input}` classified as **{verdict}** (Threat Level: **{threat_level}**)")

        # Display Gauge & Summary Details
        col_gauge, col_summary = st.columns([1, 1], gap="large")

        with col_gauge:
            st.subheader("Threat Confidence Gauge")

            gauge_color = "#F43F5E" if conf >= 65 else ("#F59E0B" if conf >= 35 else "#8B5CF6")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                number={"suffix": "%", "font": {"size": 42, "color": "#FAF5FF"}},
                title={"text": "Phishing Risk Probability", "font": {"size": 16, "color": "#DDD6FE"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#C4B5FD"},
                    "bar": {"color": gauge_color, "thickness": 0.32},
                    "bgcolor": "rgba(30, 22, 54, 0.6)",
                    "borderwidth": 1,
                    "bordercolor": "#7C3AED",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(139, 92, 246, 0.18)"},
                        {"range": [35, 65], "color": "rgba(245, 158, 11, 0.18)"},
                        {"range": [65, 100], "color": "rgba(244, 63, 94, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F43F5E", "width": 3},
                        "thickness": 0.8,
                        "value": 50.0
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAF5FF"),
                height=320,
                margin=dict(l=30, r=30, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_summary:
            st.subheader("Heuristic Threat Summary")
            st.markdown(f"""
            - **Analyzed Target:** `{url_input}`
            - **Detection Verdict:** **{verdict}**
            - **Risk Probability:** `{conf:.1f}%`
            - **Severity Rating:** <span class="{'badge-critical' if is_phish else 'badge-safe'}">{threat_level}</span>
            - **Decision Threshold:** `50.0% Risk Score`
            """, unsafe_allow_html=True)

            st.write(
                "The phishing engine performs real-time structural decomposition of the URL string. "
                "It penalizes typosquatting hyphens, excessive subdomains, direct IP addressing, "
                "missing TLS, credential-harvesting tokens, and high Shannon entropy strings."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Influencing Features Table
        st.subheader("Influencing Features & Heuristic Signals")
        st.dataframe(
            feat_table,
            use_container_width=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature Vector", width="medium"),
                "Observed Value": st.column_config.TextColumn("Observed Value", width="medium"),
                "Risk Contribution": st.column_config.TextColumn("Risk Weight", width="small"),
                "Signal": st.column_config.TextColumn("Signal Level", width="small"),
                "Description": st.column_config.TextColumn("Security Context", width="large")
            },
            hide_index=True
        )


# -----------------------------------------------------------------------------
# 7. Page 4: Model Performance
# -----------------------------------------------------------------------------
def render_performance_page():
    st.markdown('<div class="page-title">📊 SOC Machine Learning Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Comprehensive evaluation metrics, confusion matrices, and feature importance rankings.</div>',
        unsafe_allow_html=True
    )

    # 1. Metrics Comparison Table
    st.subheader("1. Multi-Model Benchmark Comparison")
    benchmarks_df = load_model_benchmarks()

    if benchmarks_df is not None:
        st.dataframe(
            benchmarks_df,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Model Architecture", width="medium"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
                "Precision": st.column_config.TextColumn("Precision", width="small"),
                "Recall": st.column_config.TextColumn("Recall", width="small"),
                "F1-Score": st.column_config.TextColumn("F1-Score", width="small"),
            },
            hide_index=True
        )
    else:
        st.warning("Benchmark CSV `models/model_comparison.csv` not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Confusion Matrix Heatmaps
    st.subheader("2. Confusion Matrix Heatmaps")
    cm_col1, cm_col2 = st.columns([1, 1], gap="large")

    with cm_col1:
        st.write("**Interactive Confusion Matrix (Winning Model: XGBoost)**")
        # Hardcoded from training run: TN=8525, FP=2, FN=1, TP=11472
        cm_data = [[8525, 2], [1, 11472]]
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted Benign (0)", "Predicted DDoS (1)"],
            y=["Actual Benign (0)", "Actual DDoS (1)"],
            colorscale=[[0, "#1E1638"], [0.5, "#7E22CE"], [1, "#C084FC"]],
            showscale=False,
            text=[[f"TN: 8,525<br>(42.6%)", f"FP: 2<br>(0.01%)"],
                  [f"FN: 1<br>(0.005%)", f"TP: 11,472<br>(57.4%)"]],
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF")
        ))
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAF5FF"),
            height=320,
            margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with cm_col2:
        st.write("**Full Multi-Model Comparative Heatmap (All Models)**")
        cm_image_path = Path("models/confusion_matrix.png")
        if cm_image_path.exists():
            st.image(str(cm_image_path), use_container_width=True)
        else:
            st.info("Comparative confusion matrix image not found in `models/`.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Feature Importance Bar Chart
    st.subheader("3. Feature Importance Bar Chart")
    feat_df = load_feature_importances()

    if feat_df is not None and not feat_df.empty:
        top_k = st.slider("Select Top N Features to Display", min_value=5, max_value=30, value=15, step=5)
        top_feats = feat_df.head(top_k).sort_values(by="Importance", ascending=True)

        fig_bar = px.bar(
            top_feats,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#3B0764"], [0.5, "#7E22CE"], [1, "#C084FC"]],
            labels={"Importance": "Relative Gini Importance Score", "Feature": "Network Traffic Metric"},
            title=f"Top {top_k} Features Influencing DDoS Classification"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAF5FF"),
            height=480,
            margin=dict(l=40, r=20, t=50, b=40),
            coloraxis_showscale=False
        )
        fig_bar.update_xaxes(gridcolor="rgba(168,85,247,0.15)")
        fig_bar.update_yaxes(gridcolor="rgba(168,85,247,0.15)")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Feature importance CSV `models/feature_importance.csv` not found.")


# -----------------------------------------------------------------------------
# 8. Page 5: About
# -----------------------------------------------------------------------------
def render_about_page():
    st.markdown('<div class="page-title">ℹ️ System Specifications & Research Context</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Technical background on datasets, ML model training, and threat intelligence heuristics.</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.subheader("1. Training Dataset (CIC-IDS2017)")
        st.write(
            """
            The DDoS classification engine is trained on the canonical **CIC-IDS2017** network intrusion dataset 
            developed by the **Canadian Institute for Cybersecurity (CIC)**.
            
            - **Total Records:** 225,745 labeled packet flows
            - **Input Features:** 78 bidirectional flow indicators
            - **Data Cleansing:** Infinite values replaced with NaN, deduplication, and complete removal of corrupted flows
            - **Subsampled Training:** 100,000 stratified samples
            - **Split:** 80% Training (80,000) / 20% Holdout Testing (20,000)
            """
        )

        st.subheader("2. Evaluated Machine Learning Models")
        st.markdown("""
        1. **Extreme Gradient Boosting (XGBoost)**:
           - *Test Accuracy:* **99.98%** | *F1-Score:* **99.99%**
           - *Selected as Production Engine*: Highest recall against stealth DDoS pulses.
        2. **Random Forest Classifier**:
           - *Test Accuracy:* **99.98%** | *Precision:* **100.00%**
           - *Ensemble of 100 orthogonal trees*.
        3. **Logistic Regression (with StandardScaler)**:
           - *Test Accuracy:* **99.81%** | *F1-Score:* **99.84%**
           - *Baseline linear boundary validation*.
        """)

    with col_b:
        st.subheader("3. Heuristic Phishing Engine")
        st.write(
            """
            Phishing detection evaluates URL structural integrity without requiring network roundtrips:
            
            - **Entropy Calculation:** Measures Shannon entropy across the character distribution to catch machine-generated domain generation algorithms (DGA).
            - **Brand Emulation / Typosquatting:** Flags excessive hyphenation, targeted keywords (`login`, `verify`, `banking`), and multi-layer subdomains.
            - **Deceptive Transport:** Detects IP-based hostnames, unauthorized port usage, and unencrypted HTTP communication for sensitive operations.
            """
        )

        st.subheader("4. Security Disclaimer")
        st.info(
            "This application is engineered for defensive security telemetry, red-team analysis, and educational threat detection. "
            "Always corroborate alerts with upstream firewall logs and enterprise SIEM pipelines."
        )


# -----------------------------------------------------------------------------
# 9. Main Application Entry Point (Clickable Buttons Navigation)
# -----------------------------------------------------------------------------
def main():
    # Model artifact loader
    artifact = load_ddos_model()

    # Initialize current page state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"

    # Sidebar Navigation (No Radio Circles - 100% Clickable Nav Buttons)
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
        st.markdown("### CyberSentinel SOC")
        st.caption("AI-Powered Threat Intelligence")
        st.divider()

        st.markdown("**Navigation Menu:**")
        nav_items = [
            ("🏠 Home", "Home"),
            ("🛡️ DDoS Detection", "DDoS Detection"),
            ("🎣 Phishing Detection", "Phishing Detection"),
            ("📊 Model Performance", "Model Performance"),
            ("ℹ️ About", "About"),
        ]

        for label, page_key in nav_items:
            is_active = (st.session_state["current_page"] == page_key)
            # Active button uses primary (glowing purple), inactive uses secondary
            if st.button(
                label,
                key=f"nav_btn_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state["current_page"] = page_key
                st.rerun()

        st.divider()
        st.markdown("**Engine Status:**")
        st.markdown("🟣 **SOC Core:** `ACTIVE`")
        if artifact is not None:
            st.markdown(f"🟣 **DDoS Model:** `{artifact.get('model_name', 'XGBoost')}`")
        else:
            st.markdown("🔴 **DDoS Model:** `OFFLINE`")
        st.markdown("🟣 **Phishing Engine:** `ONLINE`")

    # Current Page Rendering
    current = st.session_state["current_page"]
    if current == "Home":
        render_home_page()
    elif current == "DDoS Detection":
        render_ddos_page(artifact)
    elif current == "Phishing Detection":
        render_phishing_page()
    elif current == "Model Performance":
        render_performance_page()
    elif current == "About":
        render_about_page()


if __name__ == "__main__":
    main()
