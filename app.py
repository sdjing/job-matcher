import streamlit as st
import pandas as pd
import tempfile
import os
from scraper import fetch_jobs, filter_jobs, parse_jobs, save_to_csv
from parser import load_resume
from matcher import score_jobs, get_gaps
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Job Matcher", layout="wide")
st.title("Job Matcher")
st.caption("Upload your resume and match it against live job listings.")

with st.sidebar:
    st.header("Settings")
    keywords_input = st.text_input("Job keywords (comma separated)", "python, machine learning, data, ai, software engineer")
    max_jobs = st.slider("Max jobs to scan", 10, 100, 50)
    top_n = st.slider("Results to show", 5, 20, 10)
    run = st.button("Fetch & Match", type="primary")

resume_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

if run:
    if not resume_file:
        st.warning("Please upload your resume first.")
        st.stop()

    keywords = [k.strip() for k in keywords_input.split(",")]

    with st.spinner("Fetching jobs..."):
        raw_jobs = fetch_jobs()
        matched = filter_jobs(raw_jobs, keywords)
        jobs = parse_jobs(matched, max_jobs)

    if not jobs:
        st.error("No jobs matched your keywords. Try broadening them.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp:
        tmp.write(resume_file.read())
        tmp_path = tmp.name

    with st.spinner("Parsing resume..."):
        resume_text = load_resume(tmp_path)
    os.unlink(tmp_path)

    df = pd.DataFrame(jobs)
    df = df.dropna(subset=["description"])

    with st.spinner("Scoring matches..."):
        scores = score_jobs(resume_text, df["description"])
        df["match_score"] = scores
        df = df.sort_values("match_score", ascending=False).reset_index(drop=True)

        vectorizer = TfidfVectorizer(stop_words="english")
        vectorizer.fit([resume_text] + list(df["description"]))
        vocab = vectorizer.get_feature_names_out()

    st.success(f"Matched against {len(df)} jobs.")
    st.divider()

    for i, row in df.head(top_n).iterrows():
        gaps = get_gaps(resume_text, row["description"], vocab)
        score = row["match_score"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### [{row['title']}]({row['url']})")
            st.caption(f"{row['company']} · {row['location']}")
            if gaps:
                st.markdown(f"**Keyword gaps:** `{'` `'.join(gaps)}`")
            else:
                st.markdown("**Keyword gaps:** none detected")
        with col2:
            st.metric("Match Score", f"{score}%")

        st.divider()

    csv = df[["title", "company", "match_score", "url"]].to_csv(index=False)
    st.download_button("Download Results CSV", csv, "results.csv", "text/csv")