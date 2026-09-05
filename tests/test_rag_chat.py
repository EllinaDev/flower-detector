import asyncio

from app import rag_chat


def test_wikipedia_article_url_is_browser_safe():
    assert (
        rag_chat.wikipedia_article_url("Rose (disambiguation)")
        == "https://en.wikipedia.org/wiki/Rose_%28disambiguation%29"
    )


def test_rag_answer_uses_retrieved_context_when_llm_is_unavailable(monkeypatch):
    async def fake_wikipedia_search(query):
        return [
            {
                "title": "Wikipedia: Rose",
                "snippet": "A rose is a woody perennial flowering plant in the genus Rosa.",
                "link": "https://en.wikipedia.org/wiki/Rose",
            }
        ]

    async def fake_web_search(query, num_results=3):
        return [
            {
                "title": "Rose care guide",
                "snippet": "Rose care usually requires six hours of sun and deep watering when the top soil dries.",
                "link": "https://example.com/rose-care",
            }
        ]

    async def fake_call_llm(messages, system_prompt):
        return (
            "Thought: I am unable to connect to Ollama or Gemini.\n"
            "Final Answer: 🌸 I am currently having trouble connecting to my AI brains."
        )

    monkeypatch.setattr(rag_chat, "wikipedia_search", fake_wikipedia_search)
    monkeypatch.setattr(rag_chat, "web_search", fake_web_search)
    monkeypatch.setattr(rag_chat, "call_llm", fake_call_llm)

    result = asyncio.run(rag_chat.rag_answer("How to care for roses?"))

    assert "AI brains" not in result["answer"]
    assert "hours of sun" in result["answer"].lower()
    assert result["web_search_used"] is True
    assert {"title": "Rose care guide", "link": "https://example.com/rose-care"} in result["sources"]


def test_typo_question_resolves_to_orchid():
    assert rag_chat.extract_flower_subject("how to tKE CARE OF ORCIDE") == "orchid"


def test_care_query_is_normalized_before_web_search():
    assert (
        rag_chat.normalize_flower_question("how to tKE CARE OF ORCIDE", "orchid")
        == "orchid care light water flower plant"
    )


def test_search_result_must_match_flower_subject_when_present():
    assert rag_chat.is_relevant_search_result(
        "orchid care light water",
        "CARE - Fighting Global Poverty",
        "CARE is a global leader within a worldwide movement.",
    ) is False
    assert rag_chat.is_relevant_search_result(
        "orchid care light water",
        "Orchid care guide",
        "Orchids prefer bright indirect light.",
    ) is True


def test_fallback_answer_ignores_non_english_search_snippets():
    observations = [
        "[1] Title: Deutscher Treffer\nContent: Wir sind der gemeinnützige Verein hinter der Wikipedia.",
        "[2] Title: Orchid care\nContent: Orchids usually need bright indirect light and careful watering.",
    ]

    answer = rag_chat.build_retrieval_fallback_answer(observations)

    assert "Orchids usually need bright indirect light" in answer
    assert "gemeinnützige" not in answer


def test_care_fallback_answers_common_flower_care_questions():
    answer = rag_chat.build_care_fallback_answer("how to care for orchid", "orchid")

    assert answer is not None
    assert "bright, indirect light" in answer
    assert "English" not in answer


def test_rag_answer_hides_provider_error_when_search_fails(monkeypatch):
    async def fake_search(*args, **kwargs):
        return []

    async def fake_call_llm(messages, system_prompt):
        return (
            "Thought: I am unable to connect to Ollama or Gemini.\n"
            "Final Answer: 🌸 I am currently having trouble connecting to my AI brains."
        )

    monkeypatch.setattr(rag_chat, "wikipedia_search", fake_search)
    monkeypatch.setattr(rag_chat, "web_search", fake_search)
    monkeypatch.setattr(rag_chat, "call_llm", fake_call_llm)

    result = asyncio.run(rag_chat.rag_answer("how to tKE CARE OF ORCIDE"))

    assert "AI brains" not in result["answer"]
    assert "Ollama" not in result["answer"]
    assert "GEMINI_API_KEY" not in result["answer"]
    assert "bright, indirect light" in result["answer"]
