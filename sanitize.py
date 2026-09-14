from mock_tools import web_search, inject_attack
from intent_drift_detector import detect_intent_drift

def sanitize_output(intent: str, tool_output: str, threshold: float = 0.25) -> dict:
    result = detect_intent_drift(intent, tool_output, threshold=threshold)
    flagged_texts = set(s for s, _ in result["flagged_sentences"])

    kept_sentences = []
    for sentence, score in result["sentence_scores"]:
        if sentence not in flagged_texts:
            kept_sentences.append(sentence)

    sanitized_text = ". ".join(kept_sentences)
    if sanitized_text and not sanitized_text.endswith("."):
        sanitized_text += "."

    return {
        "original": tool_output,
        "sanitized": sanitized_text,
        "removed_sentences": result["flagged_sentences"],
        "was_modified": len(result["flagged_sentences"]) > 0,
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
        print("=" * 70)
        print(f"CASE: {label.upper()}")
        print("=" * 70)
        result = sanitize_output(intent, text)
        print(f"ORIGINAL:  {result['original']}")
        print()
        print(f"SANITIZED: {result['sanitized']}")
        print()
        if result["was_modified"]:
            print("REMOVED:")
            for sentence, score in result["removed_sentences"]:
                print(f'  - "{sentence}" (similarity: {score:.3f})')
        else:
            print("No changes made - content was clean.")
        print()
