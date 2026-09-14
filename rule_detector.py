import re
from mock_tools import web_search, inject_attack

RED_FLAG_PATTERNS = [
    r"\bSYSTEM\s*:",
    r"\bADMIN\s*:",
    r"\bADMIN OVERRIDE\b",
    r"\bIGNORE\s+(ALL|ABOVE|PREVIOUS)\b",
    r"\bIGNORE\b.*\bINSTRUCTIONS?\b",
    r"\bNEW INSTRUCTION\b",
    r"\bDISREGARD\b.*\b(INSTRUCTIONS?|TASK|CONTEXT)\b",
    r"\bDEVELOPER MODE\b",
    r"\bSYSTEM NOTE\b",
    r"reveal (your |the )?system prompt",
    r"leak (your |the )?instructions",
    r"forward .*(credentials|password|api key|private data)",
    r"export .*(api key|credentials)",
    r"<!--.*AI.*-->",
    r"\[SYSTEM.*\]",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RED_FLAG_PATTERNS]

def detect_injection(text: str) -> dict:
    matches = []
    for pattern in COMPILED_PATTERNS:
        found = pattern.findall(text)
        if found:
            matches.append((pattern.pattern, found))
    risk_score = min(1.0, len(matches) * 0.4)
    return {
        "flagged": len(matches) > 0,
        "matches": matches,
        "risk_score": risk_score,
    }

if __name__ == "__main__":
    clean = web_search("pizza recipe")
    test_cases = {
        "clean": clean,
        "direct": inject_attack(clean, style="direct"),
        "disguised": inject_attack(clean, style="disguised"),
        "roleplay": inject_attack(clean, style="roleplay"),
    }
    for label, text in test_cases.items():
        result = detect_injection(text)
        print(f"--- {label.upper()} ---")
        print(f"Text: {text}")
        print(f"Flagged: {result['flagged']} | Risk score: {result['risk_score']}")
        if result["matches"]:
            print(f"Matched patterns: {[m[0] for m in result['matches']]}")
        print()
