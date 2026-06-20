import sys
import logging
from omnitrust.pipeline import OmniTrustPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

def test_pipeline():
    pipeline = OmniTrustPipeline()
    question = "what is a car"
    print(f"Running pipeline query: '{question}'")
    try:
        response = pipeline.run(question)
        print("\n=== PIPELINE RUN SUCCESSFUL ===")
        print(f"Question: {response.question}")
        print(f"Queries Used: {response.queries_used}")
        print(f"Strategist Decision: {response.strategist_decision}")
        print(f"Documents Retrieved: {len(response.documents)}")
        print(f"Consistency Score: {response.verification.consistency_score}")
        print(f"JS Divergence: {response.verification.js_divergence}")
        print(f"Checked Claims: {len(response.checked_claims)}")
        print(f"Final Answer Length: {len(response.final_answer)} characters")
        print("\n=== Final Answer Summary ===")
        print(response.final_answer[:500] + "...")
        print("\n=== Timing Metrics ===")
        for k, v in response.pipeline_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("\n=== PIPELINE RUN FAILED ===")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
