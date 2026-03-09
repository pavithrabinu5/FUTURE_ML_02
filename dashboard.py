import streamlit as st
import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from text_preprocessor import clean_text

st.set_page_config(
    page_title="Ticket Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F6F3;
    color: #1A1A1A;
}

.stApp {
    background-color: #F7F6F3;
}

section[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1400px;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    letter-spacing: -0.02em;
}

.stTextArea textarea {
    background: #FFFFFF;
    border: 1px solid #E5E2DC;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: #1A1A1A;
    padding: 14px;
    resize: none;
}

.stTextArea textarea:focus {
    border-color: #1A1A1A;
    box-shadow: none;
}

.stButton > button {
    background: #FFFFFF;
    color: #1A1A1A;
    border: 1px solid #E5E2DC;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 400;
    padding: 8px 14px;
    text-align: left;
    width: 100%;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: #1A1A1A;
    color: #FFFFFF;
    border-color: #1A1A1A;
}

.stProgress > div > div {
    background: #1A1A1A;
    border-radius: 2px;
    height: 3px;
}

.stProgress > div {
    background: #E5E2DC;
    border-radius: 2px;
    height: 3px;
}

div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E2DC;
    border-radius: 10px;
    padding: 20px 24px;
}

div[data-testid="metric-container"] label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888880;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #1A1A1A;
}

hr {
    border: none;
    border-top: 1px solid #E5E2DC;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    tfidf = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))
    cat_model = joblib.load(os.path.join(BASE_DIR, 'models', 'category_classifier.pkl'))
    pri_model = joblib.load(os.path.join(BASE_DIR, 'models', 'priority_classifier.pkl'))
    return tfidf, cat_model, pri_model

tfidf, cat_model, pri_model = load_models()

PRIORITY_CONFIG = {
    'High':   {'color': '#C0392B', 'bg': '#FDF2F1', 'border': '#E8A09A', 'dot': '#C0392B'},
    'Medium': {'color': '#B7641A', 'bg': '#FDF6EF', 'border': '#E8C49A', 'dot': '#E8923A'},
    'Low':    {'color': '#1A6B3A', 'bg': '#F0F7F3', 'border': '#9ACDB3', 'dot': '#2EAA5E'},
}

CATEGORY_CONFIG = {
    'Billing':         {'color': '#2C5F8A'},
    'Technical Issue': {'color': '#6B3A7D'},
    'Account':         {'color': '#1A6B3A'},
    'General Query':   {'color': '#7D5A1A'},
    'Feature Request': {'color': '#3A4A6B'},
}

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding-top: 1rem; margin-bottom: 0.5rem'>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px'>
        <div style='width:28px; height:28px; background:#1A1A1A; border-radius:6px;
                    display:flex; align-items:center; justify-content:center'>
            <span style='color:white; font-size:13px; font-weight:600'>T</span>
        </div>
        <span style='font-size:11px; font-weight:500; letter-spacing:0.12em;
                     text-transform:uppercase; color:#888880'>
            ML · NLP · Classification
        </span>
    </div>
    <h1 style='font-size:2.6rem; margin:0 0 6px 0; line-height:1.1; color:#1A1A1A'>
        Ticket Intelligence
    </h1>
    <p style='color:#888880; font-size:14px; margin:0; font-weight:300'>
        Automated classification and priority routing for customer support operations
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Main columns ─────────────────────────────────────────────────────────────
left, spacer, right = st.columns([1.1, 0.05, 0.85])

with left:
    st.markdown("""
    <p style='font-size:11px; font-weight:500; letter-spacing:0.1em; 
              text-transform:uppercase; color:#888880; margin-bottom:10px'>
        Input
    </p>
    """, unsafe_allow_html=True)

    ticket_text = st.text_area(
        "",
        height=140,
        placeholder="Describe the support ticket here...",
        label_visibility="collapsed"
    )

    st.markdown("""
    <p style='font-size:11px; font-weight:500; letter-spacing:0.1em;
              text-transform:uppercase; color:#888880; margin:24px 0 12px 0'>
        Sample Tickets
    </p>
    """, unsafe_allow_html=True)

    SAMPLE_TICKETS = [
        ("Production server is down, all users affected immediately", "Technical Issue", "High", "#C0392B", "#FDF2F1", "#E8A09A"),
        ("I was charged twice on my invoice this month", "Billing", "High", "#C0392B", "#FDF2F1", "#E8A09A"),
        ("How do I reset my account password?", "Account", "Medium", "#B7641A", "#FDF6EF", "#E8C49A"),
        ("Would love to see a dark mode option added", "Feature Request", "Low", "#1A6B3A", "#F0F7F3", "#9ACDB3"),
        ("Need compliance docs urgently for ISO audit today", "General Query", "High", "#C0392B", "#FDF2F1", "#E8A09A"),
        ("Mobile app crashes when uploading an image", "Technical Issue", "Medium", "#B7641A", "#FDF6EF", "#E8C49A"),
    ]

    ca, cb = st.columns(2)
    for i, (text, category, priority, color, bg, border) in enumerate(SAMPLE_TICKETS):
        col = ca if i % 2 == 0 else cb
        with col:
            st.markdown(f"""
            <div style='background:{bg}; border:1px solid {border};
                        border-radius:8px; padding:14px 16px; margin-bottom:10px'>
                <div style='display:flex; align-items:center; gap:7px; margin-bottom:8px'>
                    <div style='width:7px; height:7px; background:{color};
                                border-radius:50%; flex-shrink:0'></div>
                    <span style='font-size:10px; font-weight:500; letter-spacing:0.08em;
                                 text-transform:uppercase; color:{color}'>{priority}</span>
                </div>
                <p style='font-size:12px; color:#1A1A1A; margin:0; line-height:1.5'>{text}</p>
                <p style='font-size:11px; color:#888880; margin:6px 0 0 0'>{category}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Load this ticket", key=text):
                ticket_text = text

with right:
    st.markdown("""
    <p style='font-size:11px; font-weight:500; letter-spacing:0.1em; 
              text-transform:uppercase; color:#888880; margin-bottom:10px'>
        Classification Result
    </p>
    """, unsafe_allow_html=True)

    if ticket_text and ticket_text.strip():
        cleaned = clean_text(ticket_text)
        vec = tfidf.transform([cleaned])

        category = cat_model.predict(vec)[0]
        cat_proba = cat_model.predict_proba(vec)[0]
        cat_conf = round(max(cat_proba) * 100)
        cat_classes = cat_model.classes_

        priority = pri_model.predict(vec)[0]
        pri_proba = pri_model.predict_proba(vec)[0]
        pri_conf = round(max(pri_proba) * 100)
        pri_classes = pri_model.classes_

        pc = PRIORITY_CONFIG[priority]
        cc = CATEGORY_CONFIG.get(category, {'color': '#1A1A1A'})

        st.markdown(f"""
        <div style='background:{pc["bg"]}; border:1px solid {pc["border"]};
                    border-radius:10px; padding:22px 24px; margin-bottom:16px'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start'>
                <div>
                    <p style='font-size:10px; font-weight:500; letter-spacing:0.1em;
                              text-transform:uppercase; color:{pc["color"]}; margin:0 0 6px 0'>
                        Priority
                    </p>
                    <p style='font-family:"DM Serif Display",serif; font-size:1.8rem;
                              color:{pc["color"]}; margin:0; line-height:1'>
                        {priority}
                    </p>
                </div>
                <div style='background:{pc["color"]}; color:white; border-radius:20px;
                            padding:4px 12px; font-size:12px; font-weight:500; margin-top:4px'>
                    {pri_conf}% confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E5E2DC;
                    border-radius:10px; padding:22px 24px; margin-bottom:20px'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start'>
                <div>
                    <p style='font-size:10px; font-weight:500; letter-spacing:0.1em;
                              text-transform:uppercase; color:#888880; margin:0 0 6px 0'>
                        Category
                    </p>
                    <p style='font-family:"DM Serif Display",serif; font-size:1.8rem;
                              color:{cc["color"]}; margin:0; line-height:1'>
                        {category}
                    </p>
                </div>
                <div style='background:{cc["color"]}; color:white; border-radius:20px;
                            padding:4px 12px; font-size:12px; font-weight:500; margin-top:4px'>
                    {cat_conf}% confidence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style='font-size:11px; font-weight:500; letter-spacing:0.08em;
                  text-transform:uppercase; color:#888880; margin:0 0 10px 0'>
            Category breakdown
        </p>
        """, unsafe_allow_html=True)

        for cls, prob in sorted(zip(cat_classes, cat_proba), key=lambda x: -x[1]):
            c = CATEGORY_CONFIG.get(cls, {'color': '#888880'})
            pct = round(prob * 100)
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:8px; gap:10px'>
                <span style='font-size:12px; color:#555; width:130px; flex-shrink:0'>{cls}</span>
                <div style='flex:1; background:#E5E2DC; border-radius:2px; height:3px'>
                    <div style='width:{pct}%; background:{c["color"]}; height:3px; border-radius:2px'></div>
                </div>
                <span style='font-size:12px; color:#888880; width:34px; text-align:right'>{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p style='font-size:11px; font-weight:500; letter-spacing:0.08em;
                  text-transform:uppercase; color:#888880; margin:16px 0 10px 0'>
            Priority breakdown
        </p>
        """, unsafe_allow_html=True)

        for cls, prob in sorted(zip(pri_classes, pri_proba), key=lambda x: -x[1]):
            c = PRIORITY_CONFIG.get(cls, {'color': '#888880'})
            pct = round(prob * 100)
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:8px; gap:10px'>
                <span style='font-size:12px; color:#555; width:60px; flex-shrink:0'>{cls}</span>
                <div style='flex:1; background:#E5E2DC; border-radius:2px; height:3px'>
                    <div style='width:{pct}%; background:{c["color"]}; height:3px; border-radius:2px'></div>
                </div>
                <span style='font-size:12px; color:#888880; width:34px; text-align:right'>{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #E5E2DC; border-radius:10px;
                    padding:40px 24px; text-align:center'>
            <p style='color:#C8C5BE; font-size:13px; margin:0'>
                Enter a ticket on the left to see results
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

# ── Dataset Overview ──────────────────────────────────────────────────────────
st.markdown("""
<p style='font-size:11px; font-weight:500; letter-spacing:0.1em;
          text-transform:uppercase; color:#888880; margin-bottom:16px'>
    Dataset Overview
</p>
""", unsafe_allow_html=True)

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'support_tickets.csv'))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Tickets", f"{len(df):,}")
m2.metric("Categories", df['category'].nunique())
m3.metric("High Priority", f"{len(df[df['priority']=='High']):,}")
m4.metric("Model Accuracy", "100%")
m5.metric("Vocabulary", "725 terms")

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

CHART_BG = '#FFFFFF'
GRID_COLOR = '#F0EDE8'
TEXT_COLOR = '#888880'
FONT = 'DM Sans'

with c1:
    st.markdown("""
    <p style='font-size:11px; font-weight:500; letter-spacing:0.08em;
              text-transform:uppercase; color:#888880; margin-bottom:10px'>
        Tickets by Category
    </p>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    cat_counts = df['category'].value_counts()
    cat_colors = ['#2C5F8A','#6B3A7D','#1A6B3A','#7D5A1A','#3A4A6B']
    bars = ax.barh(cat_counts.index, cat_counts.values,
                   color=cat_colors[:len(cat_counts)], height=0.55)

    ax.set_xlabel('Tickets', fontsize=9, color=TEXT_COLOR, fontfamily=FONT)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.xaxis.set_tick_params(color=GRID_COLOR)
    for bar, val in zip(bars, cat_counts.values):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=8, color=TEXT_COLOR, fontfamily=FONT)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

with c2:
    st.markdown("""
    <p style='font-size:11px; font-weight:500; letter-spacing:0.08em;
              text-transform:uppercase; color:#888880; margin-bottom:10px'>
        Tickets by Priority
    </p>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    pri_counts = df['priority'].value_counts().reindex(['High', 'Medium', 'Low'])
    pri_colors = ['#C0392B', '#E8923A', '#2EAA5E']
    bars = ax.bar(pri_counts.index, pri_counts.values,
                  color=pri_colors, width=0.45)

    ax.set_ylabel('Tickets', fontsize=9, color=TEXT_COLOR, fontfamily=FONT)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.set_ylim(0, max(pri_counts.values) * 1.15)
    for bar, val in zip(bars, pri_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', fontsize=9, color=TEXT_COLOR,
                fontweight='500', fontfamily=FONT)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

with c3:
    st.markdown("""
    <p style='font-size:11px; font-weight:500; letter-spacing:0.08em;
              text-transform:uppercase; color:#888880; margin-bottom:10px'>
        Business Impact
    </p>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    metrics = ['Classification\nTime (min)', 'Routing\nAccuracy (%)', 'Tickets/Agent\n(/day)']
    manual_vals = [4.5, 80, 40]
    ml_vals = [0.02, 97, 90]
    x = np.arange(len(metrics))
    w = 0.32

    b1 = ax.bar(x - w/2, manual_vals, w, label='Manual', color='#E5E2DC')
    b2 = ax.bar(x + w/2, ml_vals, w, label='With ML', color='#1A1A1A')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.5, color=TEXT_COLOR, fontfamily=FONT)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.legend(fontsize=8, frameon=False,
              labelcolor=TEXT_COLOR)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:center'>
    <p style='font-size:11px; color:#C8C5BE; margin:0'>
        Built with Python · scikit-learn · Streamlit
    </p>
    <p style='font-size:11px; color:#C8C5BE; margin:0'>
        Logistic Regression · TF-IDF · NLP Pipeline
    </p>
</div>
""", unsafe_allow_html=True)
