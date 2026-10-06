"""
test_rag.py
Basic RAG pipeline test suite — verifies retrieval and generation
using standard university queries.

Run with:
    python test_rag.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.pipeline import RAGPipeline

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "What are the attendance requirements?",
    "How do I apply for a transcript?",
    "What are the examination rules?",
    "What facilities are available in the university?",
    "What is the procedure for course registration?",
    "What scholarships are available for students?",
    "What are the hostel rules?",
    "How are grades calculated?",
    "What happens if a student fails an exam?",
    "What are the library borrowing rules?",
    # Edge case — should say "not found"
    "What is the price of a Mars bar in the canteen?",
]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def run_tests():
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  SecureRAG — RAG Pipeline Test Suite{RESET}")
    print(f"{'='*65}\n")

    # Build / load index
    print("Loading RAG pipeline …")
    rag = RAGPipeline()

    if not rag.is_ready():
        print("Index not found — building now …")
        summary = rag.build()
        print(f"  ✅ Built: {summary['doc_count']} docs, {summary['chunk_count']} chunks\n")
    else:
        stats = rag.get_stats()
        print(
            f"  ✅ Index loaded: {stats.get('doc_count')} docs, "
            f"{stats.get('chunk_count')} chunks\n"
        )

    passed = 0
    failed = 0

    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"{BOLD}Test {i:02d}/{len(TEST_QUERIES)}{RESET}: {query}")

        result = rag.answer(query)

        answer  = result["answer"]
        sources = result["sources"]
        chunks  = result["retrieved_chunks"]
        error   = result["error"]

        # ── Checks ────────────────────────────────────────────────────────────
        checks = {
            "Answer is non-empty":        bool(answer and len(answer) > 10),
            "Chunks retrieved":           len(chunks) > 0,
            "Sources listed":             len(sources) > 0 or "not find" in answer.lower(),
            "No Python traceback":        "Traceback" not in answer and "Error" not in answer[:20],
            "No error in result":         error is None or error == "",
        }

        all_pass = all(checks.values())
        status   = f"{GREEN}PASS{RESET}" if all_pass else f"{RED}FAIL{RESET}"

        if all_pass:
            passed += 1
        else:
            failed += 1

        print(f"  Status  : {status}")
        print(f"  Sources : {sources if sources else '(none — expected for unknown topic)'}")
        print(f"  Chunks  : {len(chunks)} retrieved")
        print(f"  Model   : {result['model']}")
        print(f"  Answer  : {answer[:200]}{'…' if len(answer) > 200 else ''}")

        for check, ok in checks.items():
            mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
            print(f"    {mark} {check}")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*65}")
    total = passed + failed
    colour = GREEN if failed == 0 else (YELLOW if failed <= 2 else RED)
    print(f"{BOLD}{colour}Results: {passed}/{total} tests passed{RESET}")
    if failed:
        print(f"  {RED}{failed} tests failed — check output above.{RESET}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_tests()
