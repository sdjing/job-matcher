import requests
import pandas as pd
import time

KEYWORDS = [
    "python", "machine learning", "data", "ai", "software",
    "developer", "engineer", "backend", "computer science", "nlp"
]
MAX_JOBS = 50
OUTPUT_FILE = "jobs.csv"


def fetch_jobs():
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0 (job-matcher-project)"}

    print("Fetching jobs from RemoteOK...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error: status code {response.status_code}")
        return []

    jobs = response.json()[1:]  # first item is metadata
    print(f"Pulled {len(jobs)} listings.")
    return jobs


def filter_jobs(jobs, keywords):
    filtered = []

    for job in jobs:
        combined = " ".join([
            job.get("position", ""),
            " ".join(job.get("tags", [])),
            job.get("description", "")
        ]).lower()

        if any(kw.lower() in combined for kw in keywords):
            filtered.append(job)

    print(f"{len(filtered)} jobs matched keywords.")
    return filtered


def parse_jobs(jobs, max_jobs):
    return [
        {
            "title":       job.get("position", "N/A"),
            "company":     job.get("company", "N/A"),
            "location":    job.get("location", "Remote"),
            "tags":        ", ".join(job.get("tags", [])),
            "description": job.get("description", ""),
            "url":         job.get("url", ""),
            "date_posted": job.get("date", ""),
        }
        for job in jobs[:max_jobs]
    ]


def save_to_csv(jobs, filename):
    df = pd.DataFrame(jobs)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} jobs to '{filename}'")
    return df


def main():
    raw_jobs = fetch_jobs()
    if not raw_jobs:
        return

    time.sleep(1)

    matched = filter_jobs(raw_jobs, KEYWORDS)
    if not matched:
        print("No matches. Try broader keywords.")
        return

    df = save_to_csv(parse_jobs(matched, MAX_JOBS), OUTPUT_FILE)
    print("\n--- Preview ---")
    print(df[["title", "company", "tags"]].head())


if __name__ == "__main__":
    main()