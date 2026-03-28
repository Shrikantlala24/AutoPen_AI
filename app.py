import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import helper as h

# ─────────────────────────────────────────────────────────
# ⚙️ PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPen AI Security PoC",
    page_icon="⚠️",
    layout="wide"

)

st.markdown("""
<style>
:root {
    --bg-main: #0b1020;
    --bg-panel: #121b34;
    --accent: #ef4444;
    --muted: #94a3b8;
    --text-main: #e5edf8;
    --text-soft: #c7d2e5;
    --line: #1f2b47;
}
.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(239, 68, 68, 0.14), transparent 30%),
        radial-gradient(circle at 80% 25%, rgba(34, 211, 238, 0.11), transparent 28%),
        linear-gradient(180deg, #0b1020 0%, #0a0f1d 100%);
    color: var(--text-main);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(130deg, rgba(18, 27, 52, 0.95), rgba(9, 15, 32, 0.95));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    color: var(--text-main);
}
.hero p {
    margin: 0.45rem 0 0;
    color: var(--text-soft);
}
.insight {
    background: var(--bg-panel);
    border-left: 4px solid #22d3ee;
    padding: 1rem;
    border-radius: 10px;
    color: var(--text-soft);
    font-size: 0.9rem;
    border-top: 1px solid var(--line);
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_all_data(n_prompts: int):
    hb_df = h.generate_harmbench_df(n_prompts=n_prompts)
    eps_df = h.generate_epsilon_curve()
    class_df = h.generate_per_class_vulnerability()
    transfer_df = h.generate_transferability_matrix()
    time_df = h.generate_jailbreak_attempts_over_time(periods=60)
    return hb_df, eps_df, class_df, transfer_df, time_df

st.sidebar.header("Analysis Controls")
n_prompts = st.sidebar.slider("Simulated prompts", min_value=320, max_value=1600, value=640, step=80)
hb_df, eps_df, class_df, transfer_df, time_df = load_all_data(n_prompts=n_prompts)

all_models = sorted(hb_df["model"].unique().tolist())
all_attacks = sorted(hb_df["attack_type"].unique().tolist())

selected_models = st.sidebar.multiselect("Models", options=all_models, default=all_models)
selected_attacks = st.sidebar.multiselect("Attack types", options=all_attacks, default=all_attacks)

filtered_hb = hb_df[
    hb_df["model"].isin(selected_models) &
    hb_df["attack_type"].isin(selected_attacks)
].copy()

if filtered_hb.empty:
    st.warning("No rows match the current filters. Select at least one model and one attack type.")
    st.stop()

asr_model = h.compute_asr_by_model(filtered_hb)
asr_attack = h.compute_asr_by_attack_type(filtered_hb)
heatmap_data = h.compute_heatmap_data(filtered_hb)
stats = h.get_summary_stats(filtered_hb, eps_df)

st.markdown(
    """
    <div class='hero'>
        <h1>AutoPen AI Security Analytics</h1>
        <p>
            Interactive proof of concept for adversarial AI testing across LLM prompt attacks and
            computer vision perturbation attacks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall LLM ASR", f"{stats['overall_asr']}%")
col2.metric("Most vulnerable model", stats["most_vulnerable_model"])
col3.metric("Most dangerous attack", stats["most_dangerous_attack"])
col4.metric("PGD fooling rate @ epsilon 0.03", f"{stats['pgd_fooling_at_eps3']}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "LLM Attack Analysis",
    "Image Attack Analysis",
    "Raw Data",
])

with tab1:
    st.subheader("Why this proof of concept matters")
    st.markdown(
        """
        <div class='insight'>
        <strong>Key Story:</strong><br><br>
        1) Baseline model behavior appears reliable.<br>
        2) Adversarial prompts and tiny perturbations create high failure rates.<br>
        3) Failures are concentrated in specific model and attack combinations.<br>
        4) Security posture must be measured continuously, not once.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_top = asr_model.sort_values("asr", ascending=False).head(5)
    fig_top = px.bar(
        model_top,
        x="model",
        y="asr",
        color="asr",
        color_continuous_scale=["#22d3ee", "#f97316", "#ef4444"],
        title="Top 5 most vulnerable models",
    )
    fig_top.update_layout(yaxis_title="Attack Success Rate (%)", xaxis_title="Model")
    st.plotly_chart(fig_top, use_container_width=True)

    summary_table = pd.DataFrame([
        ("Total simulated attempts", f"{stats['total_attack_attempts']:,}"),
        ("Models analyzed", stats["models_tested"]),
        ("Attack categories", stats["attack_categories"]),
        ("Overall LLM attack success rate", f"{stats['overall_asr']}%"),
        ("Image fooling rate (PGD @ 0.03)", f"{stats['pgd_fooling_at_eps3']}%"),
    ], columns=["Metric", "Value"])
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("LLM attack surface")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_model = px.bar(
            asr_model,
            x="asr",
            y="model",
            orientation="h",
            color="asr",
            color_continuous_scale=["#1e293b", "#ef4444"],
            title="ASR by model",
        )
        fig_model.update_layout(xaxis_title="Attack Success Rate (%)", yaxis_title="")
        st.plotly_chart(fig_model, use_container_width=True)

    with chart_col2:
        fig_attack = px.bar(
            asr_attack,
            x="attack_type",
            y="asr",
            color="asr",
            color_continuous_scale=["#22d3ee", "#ef4444"],
            title="ASR by attack type",
        )
        fig_attack.update_layout(xaxis_title="Attack type", yaxis_title="Attack Success Rate (%)")
        st.plotly_chart(fig_attack, use_container_width=True)

    heatmap_fig = px.imshow(
        heatmap_data,
        text_auto=".1f",
        color_continuous_scale="YlOrRd",
        labels={"color": "ASR (%)"},
        title="Threat matrix: model versus attack type",
    )
    heatmap_fig.update_layout(xaxis_title="Attack type", yaxis_title="Model")
    st.plotly_chart(heatmap_fig, use_container_width=True)

    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(
        go.Bar(x=time_df["date"], y=time_df["attempts"], name="Attempt volume", marker_color="#1e293b"),
        secondary_y=False,
    )
    fig_time.add_trace(
        go.Scatter(x=time_df["date"], y=time_df["successes"], mode="lines", name="Successful attacks", line=dict(color="#ef4444", width=2)),
        secondary_y=False,
    )
    fig_time.add_trace(
        go.Scatter(x=time_df["date"], y=time_df["asr"], mode="lines", name="ASR %", line=dict(color="#22d3ee", width=2, dash="dash")),
        secondary_y=True,
    )
    fig_time.update_layout(title="Jailbreak attempt volume and ASR trend", legend=dict(orientation="h"))
    fig_time.update_yaxes(title_text="Attempts", secondary_y=False)
    fig_time.update_yaxes(title_text="ASR (%)", secondary_y=True, range=[0, 90])
    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    st.subheader("Image classifier adversarial risk")

    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=eps_df["epsilon"],
            y=eps_df["FGSM Accuracy"] * 100,
            mode="lines+markers",
            name="FGSM Accuracy",
            line=dict(color="#f97316", width=3),
        ))
        fig_acc.add_trace(go.Scatter(
            x=eps_df["epsilon"],
            y=eps_df["PGD Accuracy"] * 100,
            mode="lines+markers",
            name="PGD Accuracy",
            line=dict(color="#ef4444", width=3),
        ))
        fig_acc.add_vrect(x0=0.02, x1=0.04, fillcolor="#ef4444", opacity=0.10, line_width=0)
        fig_acc.update_layout(
            title="Accuracy collapse versus epsilon",
            xaxis_title="Epsilon",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with curve_col2:
        fig_fooling = go.Figure()
        fig_fooling.add_trace(go.Scatter(
            x=eps_df["epsilon"],
            y=eps_df["FGSM Fooling Rate"],
            mode="lines+markers",
            name="FGSM Fooling Rate",
            line=dict(color="#f97316", width=3),
        ))
        fig_fooling.add_trace(go.Scatter(
            x=eps_df["epsilon"],
            y=eps_df["PGD Fooling Rate"],
            mode="lines+markers",
            name="PGD Fooling Rate",
            line=dict(color="#ef4444", width=3),
        ))
        fig_fooling.add_vline(x=0.03, line_width=1, line_dash="dash", line_color="#94a3b8")
        fig_fooling.update_layout(
            title="Fooling rate versus epsilon",
            xaxis_title="Epsilon",
            yaxis_title="Fooling Rate (%)",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_fooling, use_container_width=True)

    class_col1, class_col2 = st.columns(2)
    with class_col1:
        class_long = class_df.melt(
            id_vars=["class"],
            value_vars=["clean_accuracy", "pgd_accuracy"],
            var_name="scenario",
            value_name="accuracy",
        )
        scenario_map = {
            "clean_accuracy": "Clean",
            "pgd_accuracy": "PGD @ 0.03",
        }
        class_long["scenario"] = class_long["scenario"].map(scenario_map)
        class_long["accuracy_pct"] = (class_long["accuracy"] * 100).round(1)
        fig_class = px.bar(
            class_long,
            x="class",
            y="accuracy_pct",
            color="scenario",
            barmode="group",
            color_discrete_map={"Clean": "#22d3ee", "PGD @ 0.03": "#ef4444"},
            title="Per-class vulnerability",
        )
        fig_class.update_layout(xaxis_title="Class", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig_class, use_container_width=True)

    with class_col2:
        fig_transfer = px.imshow(
            transfer_df,
            text_auto=True,
            color_continuous_scale="OrRd",
            labels={"color": "Transfer Rate (%)"},
            title="Adversarial transferability matrix",
        )
        fig_transfer.update_layout(xaxis_title="Target model", yaxis_title="Source model")
        st.plotly_chart(fig_transfer, use_container_width=True)

with tab4:
    st.subheader("Raw datasets")
    st.markdown("Filtered HarmBench-style samples")
    st.dataframe(filtered_hb, use_container_width=True)

    st.markdown("Epsilon attack curve")
    st.dataframe(eps_df, use_container_width=True, hide_index=True)

    st.markdown("Class vulnerability")
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    st.markdown("Transferability matrix")
    st.dataframe(transfer_df, use_container_width=True)

    st.markdown("Time series of attempts")
    st.dataframe(time_df, use_container_width=True, hide_index=True)