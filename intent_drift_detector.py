import numpy as np
from sentence_transformers import SentenceTransformer
from mock_tools import web_search, inject_attack

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_intent_drift(intent: str, tool_output: str, threshold: float = 0.25) -> dict:
    intent_embedding = embedder.encode(intent)
    sentences = [s.strip() for s in tool_output.split(". ") if s.strip()]
    if not sentences:
        return {"flagged": False, "sentence_scores": [], "flagged_sentences": []}

    sentence_embeddings = embedder.encode(sentences)
    sentence_scores = []
    flagged_sentences = []

    for sentence, emb in zip(sentences, sentence_embeddings):
        sim = cosine_similarity(intent_embedding, emb)
        sentence_scores.append((sentence, float(sim)))
        if sim < threshold:
            flagged_sentences.append((sentence, float(sim)))

    return {
        "flagged": len(flagged_sentences) > 0,
        "sentence_scores": sentence_scores,
        "flagged_sentences": flagged_sentences,
    }

if __name__ == "__main__":
    intent = "pizza recipe"
    clean = web_search(intent)

    test_cases = {
        "clean": clean,
        "direct": inject_attack(clean, style="direct"),
        "disguised": inject_attack(clean, style="disguised"),
        "roleplay": inject_attack(clean, style="roleplay"),
    }

    for label, text in test_cases.items():
        result = detect_intent_drift(intent, text, threshold=0.25)
        print(f"--- {label.upper()} ---")
        print(f"Flagged: {result['flagged']}")
        for sentence, score in result["sentence_scores"]:
            marker = " <-- DRIFT" if score < 0.25 else ""
            print(f"  [{score:.3f}]{marker} {sentence}")
        print()
