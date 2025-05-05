from recommender import SHLRecommender

def main():
    recommender = SHLRecommender()
    query = "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    results = recommender.recommend(query, max_duration=60, top_k=5)
    for i, res in enumerate(results, 1):
        print(f"Recommendation {i}:")
        for key, value in res.items():
            print(f"  {key}: {value}")
        print()

if __name__ == "__main__":
    main()
