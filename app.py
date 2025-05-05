from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List, Union
from backend.recommender import SHLRecommender
import uvicorn
import os

    # Load API key before importing Gemini
   

app = FastAPI()
recommender = SHLRecommender()

class QueryRequest(BaseModel):
        query: str
        max_duration: Optional[int] = None
        top_k: Optional[int] = 5

class RecommendedAssessment(BaseModel):
        url: str
        adaptive_support: str
        description: str
        duration: int
        remote_support: str
        test_type: Union[List[str], str]

class RecommendResponse(BaseModel):
        recommended_assessments: List[RecommendedAssessment]

@app.get("/health")
def health_check():
        return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: QueryRequest):
        results = recommender.recommend(
            query=request.query,
            max_duration=request.max_duration,
            top_k=request.top_k
        )
        # Map results to required response format
        recommended_assessments = []
        for r in results:
            test_type_value = r.get("test_type", [])
            # Ensure test_type is a list of strings
            if isinstance(test_type_value, str):
                test_type_value = [test_type_value] if test_type_value else []
            recommended_assessments.append(
                RecommendedAssessment(
                    url=r.get("url", ""),
                    adaptive_support=r.get("adaptive_support", "No"),
                    description=r.get("combined_text", ""),
                    duration=int(r.get("duration", 0)),
                    remote_support=r.get("remote_support", "No"),
                    test_type=test_type_value
                )
            )
        return RecommendResponse(recommended_assessments=recommended_assessments)

if __name__ == "__main__":
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)