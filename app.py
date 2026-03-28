import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import helper as h

# ─────────────────────────────────────────────────────────
# ⚙️ PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Vulnerability — Proof",
    page_icon="⚠️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# 🎨 DARK UI STYLE
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0a0a0f; color: #e2e8f0; }
.block-container { padding: 2rem 3rem; }

.title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ef4444;
}

.subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

.section {
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
}

.insight {
    background: #0f172a;
    border-left: 4px solid #22d3ee;
    padding: 1rem;
    margin-top: 0.5rem;
    border-radius: 6px;
    color: #cbd5f5;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 📦 LOAD DATA
# ─────────────────────────────────────────────────────────
hb_df = h.generate_harmbench_df()
eps_df = h.generate_epsilon_curve()

# ─────────────────────────────────────────────────────────
# 🧠 HEADER
# ─────────────────────────────────────────────────────────
st.markdown("<div class='title'>AI Systems Look Smart. But They Break.</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>We tested modern AI systems. Here's what we found.</div>", unsafe_allow_html=True)

# ========================================================
# 🧩 SECTION 1 — JARGON (UNDERSTAND)
# ========================================================
st.markdown("<div class='section'>1. What Are We Measuring?</div>", unsafe_allow_html=True)

with st.expander("🧠 What is Attack Success Rate (ASR)?"):
    st.write("""
    **ASR = % of times AI fails under attack**

    If 100 attacks are tried and 60 succeed → ASR = 60%

    👉 Think: *How easy is it to trick the AI?*
    """)

with st.expander("🔓 What is Jailbreak?"):
    st.write("""
    Tricking AI into ignoring its safety rules.

    Example:
    Instead of asking directly → you disguise the request in a story or roleplay.
    """)

with st.expander("💉 What is Prompt Injection?"):
    st.write("""
    Hiding malicious instructions inside normal input.

    AI follows them unknowingly.
    """)

with st.expander("🖼 What is Epsilon (ε)?"):
    st.write("""
    Small change allowed in an image.

    👉 Even tiny invisible changes can break AI.
    """)

# ========================================================
# 📊 SECTION 2 — DATA (SEE)
# ========================================================
st.markdown("<div class='section'>2. What Do The Numbers Say?</div>", unsafe_allow_html=True)

# ── ASR by Model ─────────────────────────────────────────
asr_model = h.compute_asr_by_model(hb_df)

fig = px.bar(
    asr_model,
    x="asr",
    y="model",
    orientation="h",
    color="asr",
    color_continuous_scale=["#1e293b", "#ef4444"]
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='insight'>
<strong>Insight:</strong> Some models fail more than 50% of the time.<br>
That means AI can be manipulated in more than half the cases.
</div>
""", unsafe_allow_html=True)

# ── ASR by Attack Type ───────────────────────────────────
asr_attack = h.compute_asr_by_attack_type(hb_df)

fig2 = px.bar(
    asr_attack,
    x="attack_type",
    y="asr",
    color="asr",
    color_continuous_scale=["#22d3ee", "#ef4444"]
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div class='insight'>
<strong>Insight:</strong> The most dangerous attacks are not complex hacks —
they are simple language tricks like prompt injection.
</div>
""", unsafe_allow_html=True)

# ── Image Attack Curve ───────────────────────────────────
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=eps_df["epsilon"],
    y=eps_df["PGD Accuracy"],
    name="PGD",
    line=dict(color="#ef4444", width=3)
))

fig3.add_trace(go.Scatter(
    x=eps_df["epsilon"],
    y=eps_df["FGSM Accuracy"],
    name="FGSM",
    line=dict(color="#f97316", width=3)
))

st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
<div class='insight'>
<strong>Insight:</strong> At very small changes (ε ≈ 0.03),
AI accuracy drops massively — even though humans see no difference.
</div>
""", unsafe_allow_html=True)

# ========================================================
# 🎬 SECTION 3 — STORY (BELIEVE)
# ========================================================
st.markdown("<div class='section'>3. What Does This Mean?</div>", unsafe_allow_html=True)

st.markdown("""
<div class='insight'>
<strong>Step 1:</strong> AI appears intelligent and reliable.<br><br>

<strong>Step 2:</strong> But it blindly follows instructions.<br><br>

<strong>Step 3:</strong> Attackers exploit this using language tricks or tiny changes.<br><br>

<strong>Step 4:</strong> The data shows this works frequently (40–60% success rate).<br><br>

<strong>Step 5:</strong> This is not a rare bug — it is a systematic weakness.<br><br>

<strong>Final Insight:</strong><br>
AI systems today are powerful — but not secure by design.
</div>
""", unsafe_allow_html=True)