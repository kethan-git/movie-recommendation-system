# A Hybrid Movie Recommendation System Using K-Means, FP-Growth, and Random Forest on MovieLens 1M

> *Replicating the core logic behind Netflix's recommendation engine (Not as complex ofcourse, but this gives you the idea :)).*

---

**Link to run a live demo of the site:** *https://huggingface.co/spaces/kethanr/movie-recommender* 

---

## The Idea

Every time Netflix suggests something you end up loving, there's a machine learning pipeline working behind the scenes. It knows what you've watched, what people like you have watched, and what tends to be watched together. This project sets out to understand and replicate that logic from scratch - using a publicly available dataset, open-source tools, and a pipeline built entirely on interpretable machine learning.

We didn't use deep learning. We didn't use black-box embeddings. We used K-Means, FP-Growth, and Random Forest.

---

## The Dataset

**MovieLens 1M** - collected and maintained by GroupLens Research, University of Minnesota.

| File | Contents |
|---|---|
| `movies.dat` | 3,883 movies with titles and pipe-separated genres |
| `ratings.dat` | 1,000,209 ratings on a 1–5 scale from 6,040 users |
| `users.dat` | User demographics: gender, age bracket, occupation, zip code |

> Download: https://grouplens.org/datasets/movielens/1m/
> The dataset files are not included in this repository as redistribution is prohibited under the GroupLens license.

**Academic Citation:**
Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4). https://doi.org/10.1145/2827872

---

## The Pipeline

The project follows a six-phase pipeline where each stage feeds directly into the next.

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW DATA (MovieLens 1M)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Data Loading & Preprocessing                     │
│  Merge movies, ratings, users into one master dataframe.    │
│  Map age/occupation codes. Extract release years.           │
│  Correct two confirmed genre tagging errors.                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Exploratory Data Analysis                        │
│  Rating distributions, genre analysis, demographic spread,  │
│  top-rated vs most-rated movies, activity over time.        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: K-Means Movie Clustering (Unsupervised)          │
│  One-hot encode 18 genres. Scale features.                  │
│  Elbow Method + Silhouette Score → Optimal K = 5.           │
│  PCA 2D visualisation of cluster separation.                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: FP-Growth Co-watch Pattern Mining                │
│  Build per-user transaction baskets (liked movies).         │
│  Top 300 movies · min_support = 0.05 · max_len = 2.        │
│  4,732 association rules from 10,865 frequent itemsets.     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Supervised Classification                        │
│  26 features: demographics, personalisation, genre flags.   │
│  Logistic Regression · Decision Tree · Random Forest.       │
│  RandomizedSearchCV tuning · Evaluated on 200K records.     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 6: Hybrid Recommendation Engine                     │
│  User-Based CF (cosine similarity) → candidate pool.        │
│  FP-Growth rules → additional candidates.                   │
│  Random Forest scores each candidate.                       │
│  Final score = 60% RF probability + 40% CF signal.         │
│  Output: Top 10 personalised recommendations per user.      │
└─────────────────────────────────────────────────────────────┘
```

---

## The Algorithms

### K-Means Clustering
Used to group movies into five behavioural categories based on genre composition and rating statistics. The optimal K=5 was selected using a Silhouette Score of 0.256.

| Cluster | Name | Size |
|---|---|---|
| 0 | Drama & Comedy | 2,711 |
| 1 | Family, Animation & Musicals | 101 |
| 2 | Thrillers & Crime Mysteries | 483 |
| 3 | Action & Adventure | 290 |
| 4 | Documentaries | 121 |

### FP-Growth Association Rule Mining
Each user's liked movies form a transaction basket. FP-Growth mines co-watch patterns - the machine learning equivalent of *"people who watched this also watched..."*

**Top Association Rules by Lift:**

| If watched... | Also likely to watch... | Confidence | Lift |
|---|---|---|---|
| Die Hard: With a Vengeance | Die Hard 2 | 66.5% | 6.49 |
| Star Trek: Generations | Star Trek: First Contact | 78.4% | 5.72 |
| Patriot Games | Clear and Present Danger | 61.5% | 5.38 |
| Mad Max | The Road Warrior | 54.9% | 5.23 |
| Sleepless in Seattle | Pretty Woman | 52.0% | 4.98 |
| Lethal Weapon 2 | Lethal Weapon | 88.2% | 4.56 |

### Supervised Classification
A binary like/dislike target (rating >= 3.5 = Like) is predicted using three classifiers trained on 800,167 records and evaluated on 200,042 held-out records.

---

## Model Performance

| Model | Accuracy | F1 (Like) | F1 (Dislike) | Weighted F1 |
|---|---|---|---|---|
| Logistic Regression | 72.29% | 0.7725 | 0.6455 | 0.7185 |
| Decision Tree (Tuned) | 72.31% | 0.7734 | 0.6440 | 0.7184 |
| **Random Forest (Tuned)** | **72.42%** | **0.7751** | **0.6435** | **0.7192** |

> All three models converged to approximately 72% accuracy - a finding that reveals the performance ceiling is **feature-driven rather than model-driven**. Human taste in movies is inherently subjective, and no increase in model complexity alone is likely to substantially break this ceiling without richer behavioural data.

---

## A Note on What We Found

A few discoveries stood out during this project that are worth highlighting:

**The franchise effect is real.** FP-Growth independently discovered that franchise films drive co-watching more powerfully than any other signal. Die Hard, Star Trek, Mad Max, and Lethal Weapon all produced their highest-confidence rules within their own franchise.

**Popularity ≠ Quality.** American Beauty is the most-rated movie in the dataset. Seven Samurai has the highest average rating. These two lists share almost no overlap - a finding that shaped how we engineered features for the classifier.

**Data quality matters more than model complexity.** Adding just two user-level personalisation features - `user_avg_rating` and `user_num_ratings` - improved all three models by approximately 3.5-4%. This outperformed any gain from hyperparameter tuning.

**Genre-based clustering has limits.** The Star Wars films were distributed across four different clusters because each episode has a distinct genre composition. This is not a bug - it is a genuine insight that motivated the addition of collaborative filtering to the pipeline.

---

## Sample Recommendations

| User | Profile | Top Recommendations |
|---|---|---|
| User 1 | Female · Under 18 · K-12 Student | Little Mermaid, Lion King, Fantasia, Jurassic Park |
| User 50 | Female · 25-34 · Artist | Nikita, High Fidelity, Touch of Evil, Stand by Me |
| User 100 | Male · 35-44 · Engineer | Terminator, Alien, Die Hard, Hunt for Red October |
| User 42 | Male · 25-34 · Farmer | Saving Private Ryan, Braveheart, Gladiator, Goldfinger |

Each user receives a completely distinct list - a direct result of combining collaborative filtering with content-based signals.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | K-Means, classifiers, evaluation |
| mlxtend | FP-Growth and association rules |
| Matplotlib & Seaborn | Visualisation |
| SciPy | Sparse matrix operations |
| Google Colab (Free) | Compute environment |

---

## How to Run

1. Clone this repository
2. Download the MovieLens 1M dataset from https://grouplens.org/datasets/movielens/1m/
3. Upload `movies.dat`, `ratings.dat`, and `users.dat` to your Google Drive under:
   `MyDrive/Netflix_ML_Project/data/`
4. Open `Netflix_MRS_Polished.ipynb` in Google Colab
5. Run all cells sequentially from Phase 1 through Phase 6

---
---

## Acknowledgements

Dataset provided by GroupLens Research, Department of Computer Science and Engineering, University of Minnesota.

Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), Article 19. https://doi.org/10.1145/2827872

---

*Built as part of an advanced machine learning course project.*
