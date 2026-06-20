import time
import json
import logging
from omnitrust.pipeline import OmniTrustPipeline

# Suppress verbose logging to keep output readable
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluator")
logger.setLevel(logging.INFO)

TEST_QUESTIONS = [
    "What is a car?",
    "Explain the basic concept of machine learning",
    "Who discovered penicillin and when?",
    "What are the security vulnerabilities of smart contracts?",
    "What is the capital of France and what is it famous for?",
    "How does the greenhouse effect work?",
    "What is quantum entanglement?",
    "What is Retrieval-Augmented Generation (RAG)?",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?"
]

def run_evaluation():
    print("=== STARTING OMNITRUST-RAG EVALUATION ON 10 QUESTIONS ===")
    pipeline = OmniTrustPipeline()
    results = []
    
    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{idx}/10] Querying: '{question}'...")
        start_time = time.time()
        try:
            response = pipeline.run(question)
            elapsed = time.time() - start_time
            
            # Simple automatic quality metrics
            retrieved_count = len(response.documents)
            useful_count = sum(1 for d in response.documents if d.is_useful)
            consistency = response.verification.consistency_score
            claims_count = len(response.checked_claims)
            ans_len = len(response.final_answer)
            
            # Judge quality
            status = "PASS"
            reasons = []
            if retrieved_count == 0:
                status = "FAIL"
                reasons.append("No documents retrieved")
            if ans_len < 100:
                status = "FAIL"
                reasons.append("Answer too short")
            if "not be directly answered" in response.final_answer.lower() or "missing" in response.final_answer.lower():
                # A fallback answer is fine if we actually retrieved nothing, but if we did retrieve relevant docs it shouldn't say "not be directly answered"
                if retrieved_count > 3 and useful_count > 2:
                    status = "WARNING"
                    reasons.append("Answer indicates missing info despite retrieving useful documents")
            
            result_record = {
                "idx": idx,
                "question": question,
                "status": status,
                "elapsed_s": round(elapsed, 2),
                "docs_retrieved": retrieved_count,
                "docs_useful": useful_count,
                "consistency": round(consistency, 4),
                "claims_checked": claims_count,
                "answer_length": ans_len,
                "reasons": reasons,
                "answer_excerpt": response.final_answer[:200].replace('\n', ' ') + "..."
            }
            results.append(result_record)
            print(f"  Status: {status} | Time: {elapsed:.2f}s | Docs: {retrieved_count} ({useful_count} useful) | Consistency: {consistency:.2f}")
            print(f"  Excerpt: {result_record['answer_excerpt']}")
            if reasons:
                print(f"  Notes: {', '.join(reasons)}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            result_record = {
                "idx": idx,
                "question": question,
                "status": "ERROR",
                "elapsed_s": round(elapsed, 2),
                "error": str(e),
                "reasons": ["Pipeline crashed with exception"]
            }
            results.append(result_record)
            print(f"  Status: ERROR | Time: {elapsed:.2f}s")
            import traceback
            traceback.print_exc()
            
    print("\n=== EVALUATION REPORT SUMMARY ===")
    passes = sum(1 for r in results if r["status"] == "PASS")
    warnings = sum(1 for r in results if r["status"] == "WARNING")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    
    print(f"Passes: {passes} | Warnings: {warnings} | Fails: {fails} | Errors: {errors}")
    
    # Save output report
    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Report saved to eval_report.json")

if __name__ == "__main__":
    run_evaluation()
