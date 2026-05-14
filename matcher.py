import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parser import load_resume


def load_jobs(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["description"])
    return df


def score_jobs(resume_text, job_descriptions):
    corpus = [resume_text] + list(job_descriptions)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)

    resume_vector = matrix[0]
    job_vectors = matrix[1:]

    scores = cosine_similarity(resume_vector, job_vectors)[0]
    return (scores * 100).round(2)


def get_gaps(resume_text, job_description, vectorizer_vocab, top_n=10):
    job_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    vocab = set(vectorizer_vocab)

    missing = job_words & vocab - resume_words
    return list(missing)[:top_n]


def run_matcher(resume_path, jobs_path, top_n=10):
    resume_text = load_resume(resume_path)
    df = load_jobs(jobs_path)

    scores = score_jobs(resume_text, df["description"])
    df["match_score"] = scores
    df = df.sort_values("match_score", ascending=False).reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([resume_text] + list(df["description"]))
    vocab = vectorizer.get_feature_names_out()

    print(f"\n--- Top {top_n} Matches ---\n")
    for i, row in df.head(top_n).iterrows():
        gaps = get_gaps(resume_text, row["description"], vocab)
        print(f"#{i+1} {row['title']} @ {row['company']}")
        print(f"    Score : {row['match_score']}%")
        print(f"    URL   : {row['url']}")
        print(f"    Gaps  : {', '.join(gaps) if gaps else 'none detected'}")
        print()

    output = "results.csv"
    df[["title", "company", "match_score", "url"]].to_csv(output, index=False)
    print(f"Full results saved to '{output}'")


if __name__ == "__main__":
    run_matcher("resume.pdf", "jobs.csv")