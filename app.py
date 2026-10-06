"""CineMatch interactive website, powered by Streamlit."""

from __future__ import annotations

import html
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from cinematch.pipeline import artifact_path, build_model


st.set_page_config(
    page_title="CineMatch — your next favorite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --cm-bg: #0e1116;
        --cm-panel: #151a21;
        --cm-panel-2: #1a2029;
        --cm-border: rgba(255,255,255,.085);
        --cm-text: #f5f5f1;
        --cm-muted: #9aa3ad;
        --cm-accent: #c9f36a;
        --cm-accent-dark: #90bd35;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
    .stApp { background: radial-gradient(ellipse at 76% 0%, rgba(48,74,56,.19), transparent 38%), var(--cm-bg); color: var(--cm-text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #11161d; border-right: 1px solid var(--cm-border); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }
    .block-container { max-width: 1320px; padding-top: 1.6rem; padding-bottom: 2.8rem; }
    h1, h2, h3 { font-family: 'Segoe UI', system-ui, sans-serif; letter-spacing: -.035em; color: var(--cm-text); }
    .cm-topbar { display:flex; align-items:center; justify-content:space-between; padding: .2rem 0 1.4rem; border-bottom:1px solid var(--cm-border); margin-bottom:2.25rem; }
    .cm-brand { display:flex; align-items:center; gap:.65rem; color:var(--cm-text); font-family:'Segoe UI',system-ui,sans-serif; font-size:1.1rem; font-weight:800; letter-spacing:-.04em; }
    .cm-logo { display:grid; place-items:center; width:2.15rem; height:2.15rem; background:var(--cm-accent); border-radius:.75rem; color:#1b2510; font-size:1.08rem; }
    .cm-topmeta { color:var(--cm-muted); font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; }
    .cm-hero { padding:1.4rem 0 1.25rem; }
    .cm-eyebrow { color:var(--cm-accent); font-size:.73rem; font-weight:700; letter-spacing:.17em; text-transform:uppercase; margin-bottom:.8rem; }
    .cm-hero h1 { max-width:820px; font-size:clamp(2.4rem,5vw,4.35rem); line-height:1.03; margin:0 0 1rem; }
    .cm-hero h1 span { color:var(--cm-accent); }
    .cm-hero p { max-width:730px; color:#aeb5bd; font-size:1.08rem; line-height:1.7; margin:0; }
    .cm-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin:1.8rem 0 2.2rem; }
    .cm-stat { background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.018)); border:1px solid var(--cm-border); border-radius:1rem; padding:1.05rem 1.15rem; }
    .cm-stat-label { color:var(--cm-muted); font-size:.78rem; }
    .cm-stat-value { color:var(--cm-text); font-family:'Segoe UI',system-ui,sans-serif; font-size:1.45rem; font-weight:800; margin-top:.2rem; }
    .cm-section-label { color:var(--cm-accent); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:.2rem 0 .65rem; }
    .cm-section-title { font-family:'Segoe UI',system-ui,sans-serif; font-size:1.6rem; font-weight:800; letter-spacing:-.04em; margin:0 0 .35rem; }
    .cm-section-copy { color:var(--cm-muted); margin:0 0 1rem; }
    .cm-card { min-height:205px; background:linear-gradient(150deg,#1c2529,#151a21 70%); border:1px solid var(--cm-border); border-radius:1.15rem; padding:1.15rem 1.2rem; margin:.25rem 0 .8rem; box-shadow:0 12px 32px rgba(0,0,0,.13); }
    .cm-card-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:1.15rem; }
    .cm-rank { color:#78848e; font-size:.76rem; font-weight:700; letter-spacing:.12em; }
    .cm-score { color:#1a2410; background:var(--cm-accent); border-radius:999px; padding:.34rem .65rem; font-size:.76rem; font-weight:800; }
    .cm-card h3 { font-size:1.12rem; line-height:1.35; margin:0 0 .7rem; }
    .cm-tags { display:flex; flex-wrap:wrap; gap:.38rem; margin-bottom:1rem; }
    .cm-tag { color:#c0c8ce; background:rgba(255,255,255,.065); border:1px solid rgba(255,255,255,.04); border-radius:999px; padding:.25rem .52rem; font-size:.68rem; }
    .cm-reason { color:#aeb7bd; font-size:.82rem; line-height:1.45; margin:0; }
    .cm-reason strong { color:var(--cm-accent); font-weight:600; }
    .cm-empty { border:1px dashed rgba(201,243,106,.25); border-radius:1rem; padding:2rem; background:rgba(201,243,106,.025); }
    .cm-how { background:var(--cm-panel); border:1px solid var(--cm-border); border-radius:1rem; padding:1rem 1.1rem; min-height:112px; }
    .cm-how-num { color:var(--cm-accent); font-size:.72rem; letter-spacing:.1em; font-weight:800; }
    .cm-how-title { font-weight:700; margin:.35rem 0 .2rem; }
    .cm-how-copy { color:var(--cm-muted); font-size:.82rem; line-height:1.45; }
    .cm-footer { color:#7f8992; font-size:.76rem; border-top:1px solid var(--cm-border); margin-top:2.2rem; padding-top:1rem; }
    div[data-testid="stMetric"] { background:var(--cm-panel); border:1px solid var(--cm-border); border-radius:.8rem; padding:.8rem 1rem; }
    div[data-testid="stMetricLabel"] { color:var(--cm-muted); }
    .stButton > button[kind="primary"] { background:var(--cm-accent); border:0; color:#1d2713; font-weight:800; border-radius:.75rem; min-height:2.85rem; }
    .stButton > button[kind="primary"]:hover { background:#d9ff87; color:#1d2713; }
    .stButton > button:not([kind="primary"]) { border-color:var(--cm-border); border-radius:.75rem; }
    [data-testid="stTabs"] button { color:#aeb5bd; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--cm-accent); }
    [data-testid="stExpander"] { background:rgba(255,255,255,.02); border:1px solid var(--cm-border); border-radius:.8rem; }
    @media (max-width: 760px) {
        .cm-stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .cm-topmeta { display:none; }
        .cm-hero { padding-top:.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading CineMatch for the first time…")
def load_model():
    path = artifact_path()
    if not path.exists():
        return build_model()
    import joblib

    try:
        return joblib.load(path)
    except Exception as exc:
        st.warning(f"The saved model could not be loaded ({type(exc).__name__}); CineMatch is rebuilding it from the public data.")
        return build_model()


def profile_from_upload(uploaded_file, model) -> dict[int, float]:
    if uploaded_file is None:
        return {}
    try:
        payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
        raw_profile = payload.get("ratings", payload)
        profile: dict[int, float] = {}
        for movie_id, rating in raw_profile.items():
            movie_id, rating = int(movie_id), float(rating)
            if movie_id in model.movie_to_index and 0.5 <= rating <= 5:
                profile[movie_id] = rating
        return profile
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        st.sidebar.error("That file could not be read. Upload a CineMatch JSON profile.")
        return {}


def render_movie_card(row: pd.Series, rank: int) -> None:
    title = html.escape(str(row["title"]))
    year = "" if pd.isna(row["year"]) else f" · {int(row['year'])}"
    genres = [part for part in str(row["genres"]).split("|") if part and part != "(no genres listed)"]
    tags = "".join(f'<span class="cm-tag">{html.escape(genre)}</span>' for genre in genres[:4])
    score = float(row["predicted_rating"])
    reason = html.escape(str(row["reason"]))
    st.markdown(
        f"""
        <article class="cm-card">
          <div class="cm-card-top"><span class="cm-rank">PICK {rank:02d}{html.escape(year)}</span><span class="cm-score">✦ {score:.2f} / 5</span></div>
          <h3>{title}</h3>
          <div class="cm-tags">{tags}</div>
          <p class="cm-reason"><strong>Why it fits</strong> · {reason}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


model = load_model()
movies = model.movies.copy()
ratings = model.ratings

st.markdown(
    '<div class="cm-topbar"><div class="cm-brand"><span class="cm-logo">✦</span>CineMatch</div><div class="cm-topmeta">Personalized discovery · powered by MovieLens</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<section class="cm-hero"><div class="cm-eyebrow">Your personal movie guide</div><h1>A better next movie,<br><span>picked for your taste.</span></h1><p>Rate a few films you already love. CineMatch blends community ratings and genre signals to find a thoughtful next watch—and shows you why it made the list.</p></section>',
    unsafe_allow_html=True,
)

stats = [
    ("MOVIES", f"{len(movies):,}"),
    ("COMMUNITY RATINGS", f"{len(ratings):,}"),
    ("ANONYMIZED USERS", f"{ratings['userId'].nunique():,}"),
    ("GENRES", f"{len(model.genre_encoder.classes_):,}"),
]
stats_html = "".join(
    f'<div class="cm-stat"><div class="cm-stat-label">{label}</div><div class="cm-stat-value">{value}</div></div>'
    for label, value in stats
)
st.markdown(f'<div class="cm-stats">{stats_html}</div>', unsafe_allow_html=True)

title_to_id = dict(zip(movies["title"], movies["movieId"].astype(int)))
id_to_title = {movie_id: title for title, movie_id in title_to_id.items()}
with st.sidebar:
    st.markdown("## Build your taste profile")
    st.caption("Pick films you have seen and rate them. Your profile stays in this browser session unless you download it.")
    uploaded = st.file_uploader("Restore a saved profile", type=["json"])
    uploaded_profile = profile_from_upload(uploaded, model)
    prior_selected = [id_to_title[movie_id] for movie_id in uploaded_profile if movie_id in id_to_title]
    selected_titles = st.multiselect(
        "Movies you have watched",
        options=movies["title"].tolist(),
        default=prior_selected,
        placeholder="Search titles…",
        help="Choose at least two movies for personalized recommendations.",
    )
    strategy = st.selectbox("Recommendation style", ["Hybrid", "Collaborative", "Content-based", "Popular"], help="Hybrid balances similarity, genres, and community popularity.")
    recommendation_count = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if selected_titles:
    st.markdown('<div class="cm-section-label">Step 1 · Tell us what you like</div><div class="cm-section-title">Rate your movies</div><p class="cm-section-copy">Use half-star steps. Your ratings only shape this recommendation session.</p>', unsafe_allow_html=True)
    seed_rows = movies[movies["title"].isin(selected_titles)][["movieId", "title", "genres"]].copy()
    seed_rows["Your rating"] = [uploaded_profile.get(int(movie_id), 4.0) for movie_id in seed_rows["movieId"]]
    rating_table = st.data_editor(
        seed_rows,
        hide_index=True,
        use_container_width=True,
        disabled=["movieId", "title", "genres"],
        column_config={
            "movieId": None,
            "title": st.column_config.TextColumn("Movie", width="large"),
            "genres": st.column_config.TextColumn("Genres"),
            "Your rating": st.column_config.NumberColumn("Your rating", min_value=0.5, max_value=5.0, step=0.5, format="%.1f ⭐"),
        },
        key="profile_ratings",
    )
    profile = {int(row["movieId"]): float(row["Your rating"]) for _, row in rating_table.iterrows()}
else:
    profile = {}

st.markdown('<div class="cm-section-label">Step 2 · Discover something new</div>', unsafe_allow_html=True)
action_left, action_right = st.columns([1.2, 4])
with action_left:
    recommend_clicked = st.button("Build my list  →", type="primary", use_container_width=True)
with action_right:
    popular_clicked = st.button("Show community favorites", use_container_width=True)

if recommend_clicked:
    if strategy != "Popular" and len(profile) < 2:
        st.warning("Add and rate at least two movies for personalized recommendations, or choose Popular in the sidebar.")
    else:
        st.session_state["recommendations"] = model.recommend(profile, n=recommendation_count, strategy=strategy)
        st.session_state["profile"] = profile
        st.session_state["recommendation_style"] = strategy
if popular_clicked:
    st.session_state["recommendations"] = model.recommend({}, n=recommendation_count, strategy="Popular")
    st.session_state["profile"] = {}
    st.session_state["recommendation_style"] = "Popular"

recommendation_tab, insights_tab = st.tabs(["✦ Your picks", "◌ Explore the data"])
with recommendation_tab:
    if "recommendations" in st.session_state:
        recommendations = st.session_state["recommendations"]
        style = st.session_state.get("recommendation_style", "Hybrid")
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown('<div class="cm-section-label">Made for your watchlist</div><div class="cm-section-title">Your next favorites</div>', unsafe_allow_html=True)
            st.caption(f"Ranked with the {style} recommendation style. Model scores summarize preference ranking; they are not probabilities.")
        with header_right:
            st.metric("PICKS READY", f"{len(recommendations)}")
        card_columns = st.columns(2)
        for index, (_, row) in enumerate(recommendations.iterrows()):
            with card_columns[index % 2]:
                render_movie_card(row, index + 1)
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download this list as CSV",
            recommendations.to_csv(index=False).encode("utf-8"),
            file_name="cinematch_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
        saved_profile = {"ratings": {str(movie_id): rating for movie_id, rating in st.session_state.get("profile", {}).items()}}
        download_columns[1].download_button(
            "Save my taste profile",
            json.dumps(saved_profile, indent=2).encode("utf-8"),
            file_name="cinematch_profile.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="cm-empty"><div class="cm-section-title">Your watchlist starts here</div><p class="cm-section-copy">Choose a few familiar movies in the left panel, rate them, and we’ll build a tailored list. Want to browse first? Pick <b>Show community favorites</b>.</p></div>',
            unsafe_allow_html=True,
        )
        how_columns = st.columns(3)
        steps = [
            ("01", "Pick a few films", "Search for movies you know and add them to your profile."),
            ("02", "Rate what you like", "Your star ratings help the recommender learn your taste."),
            ("03", "Find your next watch", "Explore ranked picks with a short explanation for each."),
        ]
        for column, (number, title, copy) in zip(how_columns, steps):
            with column:
                st.markdown(f'<div class="cm-how"><div class="cm-how-num">{number}</div><div class="cm-how-title">{title}</div><div class="cm-how-copy">{copy}</div></div>', unsafe_allow_html=True)

with insights_tab:
    st.markdown('<div class="cm-section-label">Inside the dataset</div><div class="cm-section-title">A little movie context</div><p class="cm-section-copy">CineMatch learns patterns from public, anonymized MovieLens ratings.</p>', unsafe_allow_html=True)
    genre_counts = (
        movies.assign(genre=movies["genres"].str.split("|"))
        .explode("genre")
        .query("genre != '(no genres listed)'")
        .groupby("genre")["movieId"]
        .nunique()
        .sort_values(ascending=True)
        .rename("Movies")
        .reset_index()
    )
    chart_columns = st.columns([1.15, 1])
    with chart_columns[0]:
        st.plotly_chart(
            px.bar(genre_counts, x="Movies", y="genre", orientation="h", title="Movies by genre", color="Movies", color_continuous_scale=[[0, "#35543e"], [1, "#c9f36a"]], template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, margin=dict(l=0, r=0, t=45, b=0)),
            use_container_width=True,
        )
    with chart_columns[1]:
        st.plotly_chart(
            px.histogram(ratings, x="rating", nbins=10, title="Community rating distribution", labels={"rating": "Stars"}, color_discrete_sequence=["#c9f36a"], template="plotly_dark").update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=45, b=0)),
            use_container_width=True,
        )

st.markdown(
    '<div class="cm-footer">Data: MovieLens Latest Small · GroupLens Research, University of Minnesota. Development dataset for non-commercial use unless permission is obtained. Recommendations are generated from historical ratings and may not reflect current availability or quality.</div>',
    unsafe_allow_html=True,
)
