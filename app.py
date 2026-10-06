"""Interactive CineMatch profile and recommendation dashboard."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from cinematch.pipeline import artifact_path, build_model


st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Preparing MovieLens data and training CineMatch on first launch…")
def load_model():
    path = artifact_path()
    if not path.exists():
        return build_model()
    import joblib

    try:
        return joblib.load(path)
    except (OSError, EOFError, ValueError):
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
        st.sidebar.error("That profile file could not be read. Upload a CineMatch JSON profile.")
        return {}


model = load_model()
movies = model.movies.copy()
ratings = model.ratings

st.title("🎬 CineMatch")
st.markdown("### Find your next favorite movie")
st.write("Build a taste profile from movies you have watched. CineMatch combines community ratings and movie genres to rank personalized picks.")

metric_columns = st.columns(4)
metric_columns[0].metric("Movies", f"{len(movies):,}")
metric_columns[1].metric("Ratings", f"{len(ratings):,}")
metric_columns[2].metric("MovieLens users", f"{ratings['userId'].nunique():,}")
metric_columns[3].metric("Genres", f"{len(model.genre_encoder.classes_):,}")

with st.sidebar:
    st.header("Your profile")
    uploaded = st.file_uploader("Restore a saved profile", type=["json"])
    uploaded_profile = profile_from_upload(uploaded, model)
    title_to_id = dict(zip(movies["title"], movies["movieId"].astype(int)))
    id_to_title = {movie_id: title for title, movie_id in title_to_id.items()}
    prior_selected = [id_to_title[movie_id] for movie_id in uploaded_profile if movie_id in id_to_title]
    selected_titles = st.multiselect(
        "Search for movies you have watched",
        options=movies["title"].tolist(),
        default=prior_selected,
        placeholder="Try searching for a title…",
        help="Choose a few familiar movies, then give each a rating.",
    )
    st.caption("Your profile stays in this browser session. Download it if you want to reuse it later.")

if selected_titles:
    seed_rows = movies[movies["title"].isin(selected_titles)][["movieId", "title", "genres"]].copy()
    seed_rows["Your rating"] = [uploaded_profile.get(int(movie_id), 4.0) for movie_id in seed_rows["movieId"]]
    rating_table = st.data_editor(
        seed_rows,
        hide_index=True,
        use_container_width=True,
        disabled=["movieId", "title", "genres"],
        column_config={
            "movieId": None,
            "title": st.column_config.TextColumn("Movie"),
            "genres": st.column_config.TextColumn("Genres"),
            "Your rating": st.column_config.NumberColumn("Your rating", min_value=0.5, max_value=5.0, step=0.5, format="%.1f ⭐"),
        },
        key="profile_ratings",
    )
    profile = {
        int(row["movieId"]): float(row["Your rating"])
        for _, row in rating_table.iterrows()
    }
else:
    profile = {}

control_columns = st.columns([2, 1, 1])
with control_columns[0]:
    strategy = st.selectbox("Recommendation approach", ["Hybrid", "Collaborative", "Content-based", "Popular"])
with control_columns[1]:
    recommendation_count = st.slider("How many picks?", min_value=5, max_value=20, value=10)
with control_columns[2]:
    st.write("")
    st.write("")
    recommend_clicked = st.button("✨ Recommend movies", type="primary", use_container_width=True)

if recommend_clicked:
    if strategy != "Popular" and len(profile) < 2:
        st.warning("Add at least two movies to get personalized recommendations. You can still choose Popular to browse community favorites.")
    else:
        recommendations = model.recommend(profile, n=recommendation_count, strategy=strategy)
        st.session_state["recommendations"] = recommendations
        st.session_state["profile"] = profile

if "recommendations" in st.session_state:
    recommendations = st.session_state["recommendations"]
    st.subheader("Your recommendations")
    st.caption("Predicted preference is a ranking score based on ratings, genres, and community activity—not a guarantee or probability.")
    st.dataframe(
        recommendations[["title", "year", "genres", "predicted_rating", "reason"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "title": st.column_config.TextColumn("Movie", width="large"),
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "genres": st.column_config.TextColumn("Genres"),
            "predicted_rating": st.column_config.NumberColumn("Predicted preference", format="%.2f / 5"),
            "reason": st.column_config.TextColumn("Why it appeared", width="large"),
        },
    )
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download recommendations CSV",
        recommendations.to_csv(index=False).encode("utf-8"),
        file_name="cinematch_recommendations.csv",
        mime="text/csv",
    )
    saved_profile = {"ratings": {str(movie_id): rating for movie_id, rating in st.session_state.get("profile", {}).items()}}
    download_columns[1].download_button(
        "Save my profile JSON",
        json.dumps(saved_profile, indent=2).encode("utf-8"),
        file_name="cinematch_profile.json",
        mime="application/json",
    )
elif not profile:
    st.info("Choose a recommendation approach, then select movies and rate them in the sidebar to get your first personalized list.")

with st.expander("Explore the dataset"):
    genre_counts = (
        movies.assign(genre=movies["genres"].str.split("|"))
        .explode("genre")
        .query("genre != '(no genres listed)'")
        .groupby("genre")["movieId"]
        .nunique()
        .sort_values(ascending=False)
        .rename("Movies")
        .reset_index()
    )
    chart_columns = st.columns([3, 2])
    chart_columns[0].plotly_chart(
        px.bar(genre_counts, x="Movies", y="genre", orientation="h", title="Movies by genre", color="Movies", color_continuous_scale="Teal"),
        use_container_width=True,
    )
    chart_columns[1].plotly_chart(
        px.histogram(ratings, x="rating", nbins=10, title="Rating distribution", labels={"rating": "Stars"}, color_discrete_sequence=["#0f766e"]),
        use_container_width=True,
    )

st.divider()
st.caption("Data: MovieLens Latest Small, GroupLens Research, University of Minnesota. Development dataset; commercial use requires prior permission. See the README for attribution and limitations.")
