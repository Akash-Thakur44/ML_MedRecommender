import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


CSV_FILE = "C:/Users/AKASH/Desktop/ML_MedRecommender/drugs_side_effects_drugs_com.csv"
DB_FILE = "pharma.db"
TABLE_NAME = "medicines"

# ----------------------
# Create DB from CSV
# ----------------------
def create_db_from_csv(csv_file, db_file):
    print(f"[INFO] Creating '{db_file}' from '{csv_file}'...")
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_file)
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()
    print("[INFO] Database created successfully.")

if not os.path.exists(DB_FILE):
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file '{CSV_FILE}' not found! Please update the path.")
    create_db_from_csv(CSV_FILE, DB_FILE)
else:
    print(f"[INFO] Using existing database '{DB_FILE}'.")

# ----------------------
# Load and prepare Data
# ----------------------
def get_clean_medicine_data():
    conn=sqlite3.connect('pharma.db')
    df = pd.read_sql_query("""
        SELECT
            drug_name, medical_condition, side_effects, generic_name, drug_classes,
            brand_names, activity, rx_otc, pregnancy_category, csa, alcohol, 
            related_drugs, medical_condition_description, rating, no_of_reviews
            FROM medicines             
                           """, conn)
    conn.close()
    # Remove 'UNNAMED:' columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

ml_df = get_clean_medicine_data()

# Combined search text for text-based algorithms
ml_df['search_text'] = (
    ml_df['drug_name'].fillna('').astype(str) + ' ' +
    ml_df['medical_condition'].fillna('').astype(str) + ' ' +
    ml_df['drug_classes'].fillna('').astype(str) + ' ' +
    ml_df['side_effects'].fillna('').astype(str) + ' ' +
    ml_df['generic_name'].fillna('').astype(str)
)

# ----------------------
# TF-IDF 
# ----------------------
def build_tfidf_recommender(df, sim_weight=0.7, rating_weight=0.3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    def recommend(query, top_n=5):
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        similarity_norm = (similarities - similarities.min()) / (np.ptp(similarities) + 1e-6)
        rating = pd.to_numeric(df['rating'], errors='coerce').fillna(0).values
        rating_norm = (rating - rating.min()) / (np.ptp(rating) + 1e-6)

        overall_score = sim_weight* similarity_norm + rating_weight* rating_norm
        top_idx = overall_score.argsort()[::-1][:top_n]
        results = df.iloc[top_idx].copy()
        results['score'] = overall_score[top_idx]
        results['similarity_score'] = similarities[top_idx]
        return results[['drug_name', 'medical_condition', 'score', 'similarity_score']]
    
    return recommend

# ----------------------
# TF_IDF + Rating
# ----------------------
def build_tfidf_with_rating(df, sim_weight=0.7, rating_weight=0.3):
    vectorizer = TfidfVectorizer(stop_words = 'english')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    def recommend(query, top_n=5):
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        similarity_norm = (similarities - similarities.min()) / (np.ptp(similarities) + 1e-6)
        rating = pd.to_numeric(df['rating'], errors='coerce').fillna(0).values
        rating_norm = (rating - rating.min()) / (np.ptp(rating)+ 1e-6)

        overall_score = sim_weight * similarity_norm + rating_weight * rating_norm
        top_idxs = overall_score.argsort()[::-1][:top_n]
        results = df.iloc[top_idxs].copy()
        results['score'] = overall_score[top_idxs]
        results['similarity_score'] = similarities[top_idxs]
        return results[['drug_name', 'medical_condition', 'score', 'similarity_score']]
    
    return recommend

# ----------------------
# KNN Metadata
# ----------------------
def build_knn_metadata_recommender(df, sim_weight=0.7, rating_weight= 0.3):
    df['meta_text'] = (
        df['drug_classes'].fillna('').astype(str) + ' ' + 
        df['medical_condition'].fillna('').astype(str) + ' ' +
        df['side_effects'].fillna('').astype(str)
    )
    vectorizer = CountVectorizer(stop_words = 'english')
    meta_matrix = vectorizer.fit_transform(df['meta_text'])

    def recommend(query, top_n = 5):
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, meta_matrix).flatten()

        similarity_norm = (similarities - similarities.min()) / (np.ptp(similarities) + 1e-6)
        rating = pd.to_numeric(df['rating'], errors='coerce').fillna(0).values
        rating_norm = (rating - rating.min()) / (np.ptp(rating) + 1e-6)

        overall_score = sim_weight* similarity_norm + rating_weight* rating_norm
        top_idx = overall_score.argsort()[::-1][:top_n]
        results = df.iloc[top_idx].copy()
        results['score'] = overall_score[top_idx]
        results['similarity_score'] = similarities[top_idx]
        return results[['drug_name', 'medical_condition', 'score', 'similarity_score']]
    
    return recommend

# ----------------------
# Benchmark (Top-N results for each query)
# ----------------------
tfidf_recommender = build_tfidf_recommender(ml_df)
tfidf_rating_recommender = build_tfidf_with_rating(ml_df)
knn_recommender = build_knn_metadata_recommender(ml_df)


def benchmark_queries(user_query, top_n=5):
    results_all = []
    results_all.append({
    
            "TFIDF": tfidf_recommender(user_query, top_n)['drug_name'].tolist(),
            "TFIDF+Rating": tfidf_rating_recommender(user_query, top_n)['drug_name'].tolist(),
            "KNN": knn_recommender(user_query, top_n)['drug_name'].tolist(),
        })
    return pd.DataFrame(results_all)


algos = {
        "TFIDF": tfidf_recommender,
        "TFIDF+Rating": tfidf_rating_recommender,
        "KNN": knn_recommender
    }

print("\nAvailable models for recommendation:")
for key in algos:
    print(f"- {key}")
    
# Get user query and algorithm selection
user_query = input("\nEnter your medicine-related search query:\n> ").strip()
chosen_algo = input("Which model do you want to use? (TFIDF / TFIDF+Rating / KNN):\n> ").strip()

if chosen_algo not in algos:
    print("Invalid choice. Defaulting to TFIDF.")
    chosen_algo = "TFIDF"

n=input("\nHow many recommendations do you want: ")
top_n = int(n)

if __name__ == "__main__":

    results = algos[chosen_algo](user_query, top_n)
    print(F"\nTop {top_n} recommendations using {chosen_algo} model for query '{user_query}':\n")
    display_df = results[['drug_name', 'medical_condition', 'score', 'similarity_score']]
    print(display_df.to_string(index=False))




    
