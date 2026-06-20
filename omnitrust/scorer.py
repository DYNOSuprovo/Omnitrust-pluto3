"""
omnitrust/scorer.py — Independence and Utility scorers.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Tuple

# Try importing ML libraries, fall back to simple pure python implementations if missing
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_ML = True
except ImportError:
    HAS_ML = False

class TFIDFVectorizer:
    """Fallback TF-IDF Vectorizer implemented in pure python."""
    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.documents_count = 0

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        self.documents_count = len(documents)
        dfs = {}
        tfs = []

        # Tokenize and compute term frequencies
        for doc in documents:
            words = [w.strip(".,!?;:()\"'").lower() for w in doc.split()]
            words = [w for w in words if len(w) > 2] # simple stopword filter
            doc_tf = {}
            for w in words:
                doc_tf[w] = doc_tf.get(w, 0) + 1
            tfs.append(doc_tf)

            # Unique terms in doc
            for w in set(words):
                dfs[w] = dfs.get(w, 0) + 1

        # Build vocabulary & IDF
        vocab_idx = 0
        for w, df in dfs.items():
            self.vocabulary[w] = vocab_idx
            # Add smoothing to IDF
            self.idf[w] = math.log((self.documents_count + 1) / (df + 1)) + 1.0
            vocab_idx += 1

        # Compute TF-IDF vectors
        vectors = []
        for doc_tf in tfs:
            vec = [0.0] * len(self.vocabulary)
            for w, tf in doc_tf.items():
                if w in self.vocabulary:
                    vec[self.vocabulary[w]] = tf * self.idf[w]
            # Normalize vector (L2 norm)
            norm = math.sqrt(sum(v*v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def transform(self, query: str) -> List[float]:
        words = [w.strip(".,!?;:()\"'").lower() for w in query.split()]
        words = [w for w in words if len(w) > 2]
        query_tf = {}
        for w in words:
            query_tf[w] = query_tf.get(w, 0) + 1

        vec = [0.0] * len(self.vocabulary)
        for w, tf in query_tf.items():
            if w in self.vocabulary:
                vec[self.vocabulary[w]] = tf * self.idf[w]
        norm = math.sqrt(sum(v*v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class IndependenceScorer:
    """Scores source independence to filter redundant/copied articles."""
    def __init__(self, use_ml: bool = False):
        self.use_ml = use_ml and HAS_ML
        self.similarity_threshold = 0.85
        self.nli_threshold = 0.80

        if self.use_ml:
            try:
                self.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
                self.nli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/deberta-v3-large", num_labels=3
                )
                self.nli_model.eval()
            except Exception as e:
                print(f"[IndependenceScorer] ML model load failed: {e}. Falling back to rule-based.")
                self.use_ml = False

    def get_similarity_matrix(self, documents: List[str]) -> List[List[float]]:
        if self.use_ml:
            embeddings = self.embedder.encode(documents, normalize_embeddings=True)
            return np.dot(embeddings, embeddings.T).tolist()
        else:
            # TF-IDF Cosine Similarity fallback
            vectorizer = TFIDFVectorizer()
            vectors = vectorizer.fit_transform(documents)
            n = len(vectors)
            sim_matrix = [[0.0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    sim_matrix[i][j] = sum(vectors[i][k] * vectors[j][k] for k in range(len(vectors[i])))
            return sim_matrix

    def detect_entailment(self, premise: str, hypothesis: str) -> float:
        if self.use_ml:
            try:
                inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    logits = self.nli_model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)
                # Assuming index 2 is entailment (check label mapping in real code)
                return float(probs[0][2].item())
            except Exception:
                pass

        # Simple rule-based word overlap entailment check as fallback
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if not h_words:
            return 0.0
        overlap = p_words.intersection(h_words)
        return len(overlap) / len(h_words)

    def score_independence(self, documents: List[str]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        sim_matrix = self.get_similarity_matrix(documents)
        n = len(documents)
        assigned = [-1] * n
        clusters = []

        # Simple greedy clustering
        current_cluster = 0
        for i in range(n):
            if assigned[i] != -1:
                continue

            cluster = [i]
            assigned[i] = current_cluster

            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue

                if sim_matrix[i][j] > self.similarity_threshold:
                    entail_prob = self.detect_entailment(documents[i], documents[j])
                    if entail_prob > self.nli_threshold:
                        cluster.append(j)
                        assigned[j] = current_cluster

            clusters.append(cluster)
            current_cluster += 1

        results = []
        for i in range(n):
            cluster_id = assigned[i]
            cluster_size = len(clusters[cluster_id])
            
            # Independence is inversely proportional to cluster size
            independence_score = 1.0 / cluster_size
            is_duplicate = cluster_size > 1 and i != clusters[cluster_id][0]

            results.append({
                "doc_id": i,
                "independence_score": round(independence_score, 4),
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "is_duplicate": is_duplicate
            })
        return results


class UtilityScorer:
    """Evaluates the retrieval utility of documents."""
    def __init__(self, use_ml: bool = False):
        self.use_ml = use_ml and HAS_ML
        self.independence_scorer = IndependenceScorer(use_ml=use_ml)

    def score_utility(self, question: str, documents: List[str]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        # 1. Novelty calculation (incremental distance from prior documents)
        novelty_scores = []
        if self.use_ml:
            embeddings = self.independence_scorer.embedder.encode(documents, normalize_embeddings=True)
            for i in range(len(documents)):
                if i == 0:
                    novelty_scores.append(1.0)
                else:
                    sims = np.dot(embeddings[i], embeddings[:i].T)
                    novelty_scores.append(float(1.0 - np.max(sims)))
        else:
            # Fallback TF-IDF novelty
            vectorizer = TFIDFVectorizer()
            vectors = vectorizer.fit_transform(documents)
            for i in range(len(documents)):
                if i == 0:
                    novelty_scores.append(1.0)
                else:
                    max_sim = max(sum(vectors[i][k] * vectors[j][k] for k in range(len(vectors[i]))) for j in range(i))
                    novelty_scores.append(1.0 - max_sim)

        # 2. Contradiction checking
        contradiction_scores = [0.0] * len(documents)
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                c_score = self.detect_contradiction(documents[i], documents[j])
                contradiction_scores[i] = max(contradiction_scores[i], c_score)
                contradiction_scores[j] = max(contradiction_scores[j], c_score)

        # 3. Overall Utility formulation
        results = []
        for i in range(len(documents)):
            # Formula: U = 0.5 * Novelty + 0.3 * (1 - Contradiction) + 0.2 * length_bonus
            novelty = novelty_scores[i]
            contradiction = contradiction_scores[i]
            
            # Simple length heuristic bonus (prefer detailed summaries)
            length_bonus = min(len(documents[i].split()) / 200.0, 1.0)

            utility_score = (0.5 * novelty) + (0.3 * (1.0 - contradiction)) + (0.2 * length_bonus)

            results.append({
                "doc_id": i,
                "novelty": round(novelty, 4),
                "contradiction": round(contradiction, 4),
                "utility_score": round(utility_score, 4),
                "is_useful": utility_score >= 0.4
            })
        return results

    def detect_contradiction(self, doc1: str, doc2: str) -> float:
        """Returns contradiction probability between 0 and 1."""
        if self.use_ml:
            try:
                # Class 0: contradiction, Class 1: neutral, Class 2: entailment (standard DeBERTa)
                inputs = self.independence_scorer.nli_tokenizer(doc1, doc2, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = self.independence_scorer.nli_model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)
                return float(probs[0][0].item())
            except Exception:
                pass

        # Heuristic contradiction detection (negation word checks in overlapping text)
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        intersection = words1.intersection(words2)
        
        negations = {"not", "never", "no", "fails", "fail", "contrary", "different", "refutes", "unlike"}
        # If both contain negations or one contains negation on same nouns
        has_negation1 = len(words1.intersection(negations)) > 0
        has_negation2 = len(words2.intersection(negations)) > 0
        
        if (has_negation1 != has_negation2) and len(intersection) > 5:
            return 0.65
        return 0.15
