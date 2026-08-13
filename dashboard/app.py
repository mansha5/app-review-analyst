import os
import json
import numpy as np
import pandas as pd
import faiss
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Review Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<div style='background: linear-gradient(90deg, #1f3c6e 0%, #1565C0 100%);
     padding: 0.8rem 2rem; border-radius: 8px; margin-bottom: 1.5rem;'>
    <h2 style='color:white; margin:0; font-size:1.4rem; font-weight:700'>
        Review Intelligence Platform
    </h2>
    <p style='color:#ccd9f0; margin:0.2rem 0 0 0; font-size:0.88rem'>
        AI-powered product analytics from real user reviews &nbsp;·&nbsp;
        13 apps &nbsp;·&nbsp; 5,200+ reviews &nbsp;·&nbsp;
        Semantic search + LLM insights
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .block-container { 
        padding-top: 5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* clean metric cards */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border: 1px solid #e9ecef;
    }

    /* issue cards */
    .issue-card {
        background: #fff8f8;
        border-left: 3px solid #E53935;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.6rem;
    }
    .praise-card {
        background: #f8fff8;
        border-left: 3px solid #2E7D32;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.6rem;
    }
    .bug-card {
        background: #fff8f0;
        border-left: 3px solid #F57C00;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.6rem;
    }
    .feature-card {
        background: #f0f4ff;
        border-left: 3px solid #1565C0;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.6rem;
    }

    /* ai answer box */
    .ai-answer {
        background: #f0f4ff;
        border: 1px solid #c5d3f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    /* sidebar */
    div[data-testid="stSidebarContent"] {
        background: #f8f9fa;
    }
    .app-health-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 0.85rem;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ── Category colors ───────────────────────────────────
CATEGORY_COLORS = {
    "complaint":       "#E53935",
    "bug":             "#F57C00",
    "feature_request": "#1565C0",
    "praise":          "#2E7D32",
    "other":           "#546E7A",
}

CARD_CLASS = {
    "complaint":       "issue-card",
    "bug":             "bug-card",
    "feature_request": "feature-card",
    "praise":          "praise-card",
    "other":           "issue-card",
}

COMPETITOR_PAIRS = {
    "zomato":    "swiggy",
    "swiggy":    "zomato",
    "paytm":     "phonepe",
    "phonepe":   "paytm",
    "instagram": "youtube",
    "youtube":   "instagram",
    "spotify":   "netflix",
    "netflix":   "spotify",
    "uber":      "swiggy",
    "airbnb":    "uber",
    "cred":      "paytm",
    "duolingo":  "instagram",
    "whatsapp":  "instagram",
}

# ── Loaders ───────────────────────────────────────────
@st.cache_data
def load_analytics():
    with open("analysis/all_analytics.json") as f:
        return json.load(f)

@st.cache_data
def load_reviews(app_name):
    return pd.read_parquet(f"data/processed/{app_name}.parquet")

@st.cache_resource
def load_rag(app_name):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(f"vectors/{app_name}.index")
    df    = pd.read_parquet(f"data/processed/{app_name}.parquet")
    return model, index, df

# ── RAG ───────────────────────────────────────────────
def retrieve_reviews(question, model, index, df, top_k=15):
    q_vec = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    _, indices = index.search(q_vec, top_k)
    return df.iloc[indices[0]]

def ask_analyst(question, app_name, retrieved_df, analytics_data):
    reviews_text = "\n".join(
        f"[{row.rating}★] {row.review_text}"
        for _, row in retrieved_df.iterrows()
    )
    high_issues = analytics_data.get("high_priority_issues", [])
    issues_text = "\n".join(
        f"- {i['label']} ({i['review_count']} reviews, "
        f"rating {i['avg_rating']}): {i.get('root_cause','')}"
        for i in high_issues
    )
    prompt = f"""You are a senior product analyst for {app_name}.
Answer the PM's question using only the evidence below.
Cite specific reviews with their star ratings.
Be concise, analytical, and actionable.

HIGH PRIORITY ISSUES:
{issues_text}

RELEVANT REVIEWS (retrieved via semantic search):
{reviews_text}

PM QUESTION: {question}

Structure:
1. Direct answer (1-2 sentences)
2. Evidence — cite 2-3 specific reviews with ratings
3. Recommended action (1 sentence)"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

# ── Weekly trend helper ───────────────────────────────
def build_weekly_trend(df):
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["week"] = df["review_date"].dt.to_period("W").apply(
        lambda r: r.start_time
    )
    weekly = (
        df.groupby("week")
        .agg(
            review_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("vader_score", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("week")
    )
    return weekly

# ── Load data ─────────────────────────────────────────
analytics = load_analytics()
apps      = sorted(analytics.keys())

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## Review Intelligence")
    st.markdown("---")
    selected_app = st.selectbox("Select App", apps)
    data         = analytics[selected_app]

    st.markdown("---")
    st.markdown("**Portfolio Health**")
    st.markdown("<div style='font-size:0.8rem; color:#888; margin-bottom:6px'>sorted by avg rating</div>", unsafe_allow_html=True)

    sorted_apps = sorted(apps, key=lambda a: analytics[a]["avg_rating"])
    for app in sorted_apps:
        r = analytics[app]["avg_rating"]
        n = analytics[app]["pct_negative"]
        dot = (
            "🔴" if r < 3 else
            "🟡" if r < 3.8 else
            "🟢"
        )
        weight = "font-weight:600" if app == selected_app else ""
        st.markdown(
            f"<div class='app-health-row' style='{weight}'>"
            f"<span>{dot} {app.title()}</span>"
            f"<span style='color:#555'>{r} &nbsp;·&nbsp; "
            f"<span style='color:#e53935'>{n}% neg</span></span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ── Header ────────────────────────────────────────────
avg    = data["avg_rating"]
color  = "#E53935" if avg < 3 else "#F57C00" if avg < 3.8 else "#2E7D32"

hc1, hc2 = st.columns([3, 1])
with hc1:
    st.title(selected_app.title())
    st.caption(f"Review Intelligence Report  ·  {data['total_reviews']} reviews analyzed")
with hc2:
    st.markdown(
        f"<div style='text-align:right; padding-top:0.8rem'>"
        f"<span style='font-size:2.8rem; font-weight:700; color:{color}'>{avg}</span>"
        f"<span style='color:#888; font-size:1rem'> / 5.0</span>"
        f"</div>",
        unsafe_allow_html=True
    )

# ── KPI Row ───────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Rating",           f"{data['avg_rating']}")
k2.metric("Positive Reviews",     f"{data['pct_positive']}%")
k3.metric("Negative Reviews",     f"{data['pct_negative']}%")
k4.metric("Neutral Reviews",      f"{data['pct_neutral']}%")
k5.metric("High Priority Issues", len(data["high_priority_issues"]))

st.markdown("---")

# ══════════════════════════════════════════════════════
# AI ANALYST
# ══════════════════════════════════════════════════════

st.markdown("#### Ask the AI Product Analyst")
st.caption("Answers are grounded in actual reviews via semantic search.")

# reset on app switch
if st.session_state.get("last_app") != selected_app:
    st.session_state.ai_question = ""
    st.session_state.ai_answer   = ""
    st.session_state.last_app    = selected_app

# suggested questions
sq1, sq2, sq3, sq4, sq5 = st.columns(5)
suggestions = {
    sq1: "Why are users unhappy?",
    sq2: "What features are requested?",
    sq3: "What should we prioritize?",
    sq4: "What are users praising?",
    sq5: "What changed recently?",
}
for col, q in suggestions.items():
    if col.button(q, use_container_width=True):
        st.session_state.ai_question = q
        st.session_state.ai_answer   = ""

# text input + button on same row
qi1, qi2 = st.columns([5, 1])
with qi1:
    typed = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder=f"Ask anything about {selected_app} reviews...",
        value=st.session_state.get("ai_question", ""),
        key="ai_text_input"
    )
with qi2:
    ask_clicked = st.button("Ask", use_container_width=True, type="primary")

if ask_clicked and typed.strip():
    st.session_state.ai_question = typed.strip()
    st.session_state.ai_answer   = ""

# show answer directly below
if st.session_state.get("ai_question") and not st.session_state.get("ai_answer"):
    q = st.session_state.ai_question
    model_rag, index_rag, df_rag = load_rag(selected_app)
    with st.spinner("Retrieving relevant reviews..."):
        retrieved = retrieve_reviews(q, model_rag, index_rag, df_rag)
        answer    = ask_analyst(q, selected_app, retrieved, data)
    st.session_state.ai_answer  = answer
    st.session_state.ai_last_q  = q

if st.session_state.get("ai_answer"):
    st.markdown(
        f"<div class='ai-answer'>"
        f"<div style='font-size:0.8rem; color:#555; margin-bottom:6px'>"
        f"Q: <em>{st.session_state.ai_question}</em></div>"
        f"{st.session_state.ai_answer}"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ══════════════════════════════════════════════════════
# TABS — Analytics + Competitor
# ══════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Analytics", "Competitor Comparison"])

# ════════════════════════════════════════════════════
# TAB 1 — Analytics
# ════════════════════════════════════════════════════
with tab1:

    # Row 1: weekly trend + rating distribution
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Rating Trend by Week")
        df_reviews = load_reviews(selected_app)
        weekly     = build_weekly_trend(df_reviews)

        if len(weekly) >= 2:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=weekly["week"],
                y=weekly["avg_rating"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.12)",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=6, color="#1f77b4"),
                name="Avg Rating",
                hovertemplate=(
                    "<b>Week of %{x|%b %d}</b><br>"
                    "Avg Rating: %{y:.2f}<extra></extra>"
                )
            ))
            fig_trend.update_layout(
                height=300,
                margin=dict(t=10, b=20, l=10, r=10),
                yaxis=dict(range=[1, 5], gridcolor="#f0f0f0"),
                xaxis=dict(showgrid=False),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data across weeks for trend analysis.")

    with col2:
        st.subheader("Rating Distribution")
        rating_dist = data["rating_distribution"]
        all_stars   = [1, 2, 3, 4, 5]
        star_colors = {
            1: "#E53935",
            2: "#F57C00",
            3: "#FDD835",
            4: "#7CB342",
            5: "#2E7D32",
        }

        rating_dist_int = {int(k): v for k, v in rating_dist.items()}
        y_vals = [rating_dist_int.get(s, 0) for s in all_stars]

        fig_dist = go.Figure(go.Bar(
            x=["1★", "2★", "3★", "4★", "5★"],
            y=y_vals,
            marker_color=[star_colors[s] for s in all_stars],
            hovertemplate="<b>%{x} stars</b><br>%{y} reviews<extra></extra>",
        ))
        fig_dist.update_layout(
            height=300,
            margin=dict(t=10, b=20, l=10, r=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="Stars", showgrid=False),
            yaxis=dict(title="Reviews", gridcolor="#f0f0f0"),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # Row 2: donut + high priority issues
    col3, col4 = st.columns([1, 2])

    with col3:
        st.subheader("Issue Breakdown")
        cat = data.get("category_breakdown", {})
        if cat:
            labels = list(cat.keys())
            values = list(cat.values())
            colors = [CATEGORY_COLORS.get(l, "#888") for l in labels]

            fig_donut = go.Figure(go.Pie(
                labels=[l.replace("_", " ").title() for l in labels],
                values=values,
                hole=0.48,
                marker=dict(
                    colors=["#66C2A5","#FC8D62","#8DA0CB","#E78AC3","#A6D854"],
                    line=dict(color="white", width=2)
                ),
                textposition="inside",
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{value} reviews · %{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                height=340,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    x=1.0,
                    y=0.5,
                    font=dict(size=12)
                ),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with col4:
        st.subheader("High Priority Issues")
        issues = data.get("high_priority_issues", [])
        if issues:
            for issue in issues:
                card = CARD_CLASS.get(issue["category"], "issue-card")
                st.markdown(
                    f"<div class='{card}'>"
                    f"<strong>{issue['label']}</strong>"
                    f"<span style='float:right; color:#888; font-size:0.82rem'>"
                    f"{issue['review_count']} reviews · "
                    f"avg {issue['avg_rating']} stars</span><br>"
                    f"<span style='font-size:0.88rem; color:#444'>"
                    f"<strong>Root cause:</strong> "
                    f"{issue.get('root_cause', '—')}</span><br>"
                    f"<span style='font-size:0.88rem; color:#444'>"
                    f"<strong>PM action:</strong> "
                    f"{issue.get('pm_action', '—')}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.success("No high priority issues detected.")

    st.markdown("---")

    # Row 3: review browser
    st.subheader("Review Browser")
    fc1, fc2 = st.columns(2)
    with fc1:
        rating_filter = st.multiselect(
            "Filter by rating",
            options=[1, 2, 3, 4, 5],
            default=[1, 2],
        )
    with fc2:
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            options=["negative", "neutral", "positive"],
            default=["negative"],
        )

    filtered = df_reviews.copy()
    if rating_filter:
        filtered = filtered[filtered["rating"].isin(rating_filter)]
    if sentiment_filter:
        filtered = filtered[filtered["vader_sentiment"].isin(sentiment_filter)]

    st.dataframe(
        filtered[[
            "review_date", "rating", "vader_sentiment",
            "vader_score", "review_text"
        ]]
        .sort_values("review_date", ascending=False)
        .head(100)
        .rename(columns={
            "review_date":     "Date",
            "rating":          "Stars",
            "vader_sentiment": "Sentiment",
            "vader_score":     "Score",
            "review_text":     "Review",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ════════════════════════════════════════════════════
# TAB 2 — Competitor Comparison
# ════════════════════════════════════════════════════
with tab2:
    competitor = COMPETITOR_PAIRS.get(selected_app)

    if not competitor:
        st.info("No competitor mapping defined for this app.")
    else:
        comp_data = analytics[competitor]
        st.subheader(f"{selected_app.title()}  vs  {competitor.title()}")
        st.caption("Side-by-side product health comparison")
        st.markdown("---")

        def delta(a, b): return round(a - b, 2)

        # labeled KPI columns
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown(f"**{selected_app.title()}**")
            m1, m2 = st.columns(2)
            m1.metric("Avg Rating",       f"{data['avg_rating']}")
            m2.metric("Negative Reviews", f"{data['pct_negative']}%")
            m3, m4 = st.columns(2)
            m3.metric("Positive Reviews", f"{data['pct_positive']}%")
            m4.metric("High Priority",    len(data['high_priority_issues']))

        with ac2:
            st.markdown(f"**{competitor.title()}**")
            m5, m6 = st.columns(2)
            m5.metric(
                "Avg Rating",
                f"{comp_data['avg_rating']}",
                delta=delta(comp_data['avg_rating'], data['avg_rating']),
            )
            m6.metric(
                "Negative Reviews",
                f"{comp_data['pct_negative']}%",
                delta=delta(comp_data['pct_negative'], data['pct_negative']),
                delta_color="inverse",
            )
            m7, m8 = st.columns(2)
            m7.metric(
                "Positive Reviews",
                f"{comp_data['pct_positive']}%",
                delta=delta(comp_data['pct_positive'], data['pct_positive']),
            )
            m8.metric(
                "High Priority",
                len(comp_data['high_priority_issues']),
                delta=(
                    len(comp_data['high_priority_issues'])
                    - len(data['high_priority_issues'])
                ),
                delta_color="inverse",
            )

        st.markdown("---")

        # rating distribution comparison
        st.subheader("Rating Distribution")
        r1 = data["rating_distribution"]
        r2 = comp_data["rating_distribution"]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name=selected_app.title(),
            x=["1★", "2★", "3★", "4★", "5★"],
            y=[r1.get(s, r1.get(str(s), 0)) for s in [1,2,3,4,5]],
            marker_color="#1f77b4",
        ))
        fig_comp.add_trace(go.Bar(
            name=competitor.title(),
            x=["1★", "2★", "3★", "4★", "5★"],
            y=[r2.get(s, r2.get(str(s), 0)) for s in [1,2,3,4,5]],
            marker_color="#E53935",
        ))
        fig_comp.update_layout(
            barmode="group",
            height=320,
            margin=dict(t=10, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Reviews", gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # sentiment comparison
        st.subheader("Sentiment Breakdown")
        sent_fig = go.Figure()
        for app_n, app_d, col in [
            (selected_app.title(), data,      "#1f77b4"),
            (competitor.title(),   comp_data, "#E53935"),
        ]:
            sent_fig.add_trace(go.Bar(
                name=app_n,
                x=["Positive", "Neutral", "Negative"],
                y=[app_d["pct_positive"],
                   app_d["pct_neutral"],
                   app_d["pct_negative"]],
                marker_color=col,
            ))
        sent_fig.update_layout(
            barmode="group",
            height=300,
            margin=dict(t=10, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(title="%", gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(sent_fig, use_container_width=True)

        st.markdown("---")

        # top issues side by side
        st.subheader("Top Issues")
        ic1, ic2 = st.columns(2)

        with ic1:
            st.markdown(f"**{selected_app.title()}**")
            app_issues = data["high_priority_issues"]
            if app_issues:
                for issue in app_issues[:5]:
                    card = CARD_CLASS.get(issue["category"], "issue-card")
                    st.markdown(
                        f"<div class='{card}'>"
                        f"<strong>{issue['label']}</strong><br>"
                        f"<span style='font-size:0.82rem; color:#666'>"
                        f"{issue['review_count']} reviews · "
                        f"avg {issue['avg_rating']} stars</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.success("No high priority issues.")

        with ic2:
            st.markdown(f"**{competitor.title()}**")
            comp_issues = comp_data["high_priority_issues"]
            if comp_issues:
                for issue in comp_issues[:5]:
                    card = CARD_CLASS.get(issue["category"], "issue-card")
                    st.markdown(
                        f"<div class='{card}'>"
                        f"<strong>{issue['label']}</strong><br>"
                        f"<span style='font-size:0.82rem; color:#666'>"
                        f"{issue['review_count']} reviews · "
                        f"avg {issue['avg_rating']} stars</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.success("No high priority issues.")