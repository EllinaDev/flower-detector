"""
rag_chat.py — Agentic RAG (Retrieval-Augmented Generation) chatbot engine for BloomBot.

Features:
  1. Ollama Support: Connects to a local Ollama server (defaults to port 11434).
  2. Auto-Model Detection: Automatically lists running Ollama models and uses the first available one.
  3. Resilient Fallback: Automatically falls back to Gemini API (using fallback models) if Ollama is unavailable.
  4. Agentic ReAct Loop: An LLM-guided agent that decides to call web search (DuckDuckGo) or Wikipedia dynamically.
  5. Completely Grounded: No hardcoded internal database is used; all answers are derived from live search results.
"""

import os
import logging
import asyncio
import httpx
import re
import difflib
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

# Ensure .env is loaded even if this module is imported standalone
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

FLOWER_ALIASES = {
    "orcide": "orchid",
    "orchide": "orchid",
    "orkide": "orchid",
    "orcid": "orchid",
    "roses": "rose",
    "orchids": "orchid",
    "sunflowers": "sunflower",
}

SEARCH_FLOWER_TERMS = [
    "rose", "orchid", "sunflower", "tulip", "hibiscus", "lavender", "daisy",
    "lily", "jasmine", "peony", "daffodil", "hydrangea", "marigold",
    "carnation", "chrysanthemum", "geranium", "iris", "poppy", "violet",
    "osteospermum",
]

CARE_FALLBACKS = {
    "orchid": "Keep orchids in bright, indirect light, water when the potting mix is nearly dry, and let extra water drain completely. Use an airy orchid mix, avoid leaving roots in standing water, and keep them away from cold drafts. 🌸",
    "rose": "Give roses at least 6 hours of sun, plant them in well-draining soil, and water deeply at the base when the top soil starts to dry. Remove dead blooms and prune weak or crowded stems to keep air moving around the plant. 🌹",
    "sunflower": "Sunflowers need full sun, steady watering while young, and well-draining soil. Once established, water deeply rather than little and often, and support tall varieties if they start to lean. 🌻",
}

QUESTION_STOPWORDS = [
    "how to care for", "how to take care of", "take care of", "how to grow",
    "how to water", "care for", "care", "grow", "watering", "water",
    "how to", "best way to", "guide", "tips", "planting", "plant",
    "often", "how", "often to", "should i", "tell me about", "what is",
    "tke", "take", "of",
]

# Intent detection: maps keyword patterns to a descriptive search topic
QUESTION_INTENT_MAP = [
    (["how to care", "take care", "tke care", "care for", "caring"], "care guide tips"),
    (["how to water", "watering", "how often water", "when to water"], "watering guide how often"),
    (["how to grow", "growing", "how to plant", "planting"], "growing planting guide"),
    (["where does", "where do", "where is", "native to", "habitat", "origin", "where grow", "where found", "where can"], "native habitat where it grows"),
    (["how much sun", "sunlight", "light requirement", "full sun", "shade"], "sunlight requirements"),
    (["fertilizer", "fertilize", "feeding", "feed"], "fertilizer feeding guide"),
    (["prune", "pruning", "trim", "trimming"], "pruning trimming guide"),
    (["soil", "potting", "pot", "repot"], "soil potting requirements"),
    (["bloom", "flowering season", "when does", "flower season"], "blooming season when it flowers"),
    (["indoor", "inside", "houseplant"], "growing indoors houseplant guide"),
    (["outdoor", "outside", "garden"], "outdoor growing garden guide"),
    (["disease", "pest", "bug", "insect", "problem", "dying"], "common diseases pests problems"),
    (["toxic", "poison", "safe for", "pet"], "toxicity safe for pets"),
]


def detect_question_intent(question: str) -> str:
    """Detect the user's specific intent from their question and return a descriptive topic suffix."""
    q_lower = question.lower()
    for patterns, topic in QUESTION_INTENT_MAP:
        if any(p in q_lower for p in patterns):
            return topic
    return ""  # No specific intent detected — general question


def build_targeted_search_query(question: str, flower_subject: str) -> str:
    """Build a focused search query combining the flower name and the user's specific intent."""
    intent_topic = detect_question_intent(question)
    if flower_subject and intent_topic:
        return f"{flower_subject} {intent_topic}"
    elif flower_subject:
        return f"{flower_subject} flower"
    return question

NON_ENGLISH_MARKERS = [
    " wir ", " der ", " die ", " und ", " ist ", " eine ", "verein",
    "enzyklopadie", "enzyklopädie", "gemeeinnutzige", "gemeinnützige",
    " nasil ", " için ", " nedir ", " bakim ", " bakım ",
]


def wikipedia_article_url(title: str) -> str:
    """Build a browser-safe Wikipedia article URL from an article title."""
    slug = quote(title.replace(" ", "_"), safe="")
    return f"https://en.wikipedia.org/wiki/{slug}"


def is_probably_english(text: str) -> bool:
    """Keep fallback snippets in English so localized search results do not leak into chat."""
    normalized = f" {text.lower()} "
    if any(marker in normalized for marker in NON_ENGLISH_MARKERS):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    ascii_letters = [ch for ch in letters if ord(ch) < 128]
    return len(ascii_letters) / len(letters) >= 0.94


def normalize_flower_question(question: str, flower_subject: str) -> str:
    """Clean user typos before sending the query to web search."""
    q_lower = question.lower()
    if any(word in q_lower for word in ["care", "tke", "take", "water", "grow"]):
        return f"{flower_subject} care light water flower plant"
    if flower_subject and flower_subject not in q_lower:
        return f"{question} {flower_subject}"
    return question


def is_care_question(question: str) -> bool:
    q_lower = question.lower()
    return any(word in q_lower for word in ["care", "tke", "take", "water", "grow"])


def build_care_fallback_answer(question: str, flower_subject: str) -> Optional[str]:
    """Provide practical English care help when no configured LLM is available."""
    if not is_care_question(question):
        return None
    return CARE_FALLBACKS.get(flower_subject)


def is_relevant_search_result(query: str, title: str, snippet: str) -> bool:
    """Reject generic/localized results that do not mention the requested flower."""
    query_lower = query.lower()
    required_terms = [term for term in SEARCH_FLOWER_TERMS if term in query_lower]
    if not required_terms:
        return True

    text = f"{title} {snippet}".lower()
    return any(term in text or f"{term}s" in text for term in required_terms)

# ─── LLM Config Constants ──────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "llama3").strip()
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

# ReAct loop configuration
MAX_ITERATIONS = 3

# Global HTTP client
_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    """Get or create a persistent asynchronous HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=15.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            verify=True
        )
    return _http_client

# ─── Live Search Tools ───────────────────────────────────────────────────────

async def web_search(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Searches the web via DuckDuckGo using the ddgs library.
    Returns:
        List of dicts with keys: title, snippet, link
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("ddgs / duckduckgo-search library not found.")
            return []

    # Clean search query: remove forced "English" suffixes that break DDGS matching
    search_query = query.strip()
    for drop_word in [" English", " english"]:
        search_query = search_query.replace(drop_word, "")

    query_lower = search_query.lower()
    if not any(w in query_lower for w in ["flower", "plant", "garden", "botan", "soil", "care", "grow", "rose", "orchid", "sunflower", "tulip", "daisy"]):
        search_query = f"{search_query} plant"

    logger.info(f"Executing web search for clean query: {search_query!r}")

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    def _search():
        try:
            with DDGS(timeout=6) as ddgs:
                search_results = list(ddgs.text(
                    search_query,
                    max_results=num_results * 2,
                    safesearch="moderate"
                ))
                if not search_results:
                    return []
                
                parsed = []
                for r in search_results:
                    title = r.get("title", "").strip()
                    snippet = r.get("body", "").strip()
                    text_content = snippet if snippet else title
                    if not text_content:
                        continue
                    if not is_probably_english(f"{title} {text_content}"):
                        continue
                    if not is_relevant_search_result(search_query, title, text_content):
                        continue
                    parsed.append({
                        "title": title[:80],
                        "snippet": text_content[:300],
                        "link": r.get("href", "")
                    })
                    if len(parsed) >= num_results:
                        break
                return parsed
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    results = await loop.run_in_executor(executor, _search)
    return results[:num_results]


async def wikipedia_search(query: str) -> List[Dict[str, str]]:
    """
    Searches Wikipedia using the REST and MediaWiki Query APIs.
    Automatically cleans query of action/question words to isolate the plant subject.
    Returns:
        List of dicts with keys: title, snippet, link
    """
    # Clean the search query of common care/action words to focus on the flower subject noun
    clean_query = query.lower()
    for word in ["how to care for", "how to grow", "how to water", "care for", "care", "grow", "watering", "water", "how to", "best way to", "guide", "tips", "planting", "plant"]:
        clean_query = clean_query.replace(word, "")
    clean_query = clean_query.strip()
    if not clean_query:
        clean_query = query

    logger.info(f"Executing Wikipedia search for cleaned subject: {clean_query!r} (original: {query!r})")
    client = await get_http_client()
    
    # Wikipedia API strictly requires a descriptive User-Agent header to prevent 403 Forbidden blocks
    headers = {
        "User-Agent": "BloomBot/1.0 (contact@bloomiq.org) httpx/0.24.0"
    }
    
    try:
        # Step 1: Search Wikipedia for matching articles
        search_url = f"https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "format": "json",
            "origin": "*"
        }
        resp = await client.get(search_url, params=params, headers=headers)
        if resp.status_code != 200:
            return []
            
        search_data = resp.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []

        # Get top 2 results
        parsed_results = []
        for result in search_results[:2]:
            title = result.get("title", "")
            pageid = result.get("pageid", "")
            
            # Step 2: Fetch detailed introduction extract from MediaWiki API
            extract_url = "https://en.wikipedia.org/w/api.php"
            extract_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": "1",
                "explaintext": "1",
                "titles": title,
                "format": "json",
                "origin": "*"
            }
            extract_resp = await client.get(extract_url, params=extract_params, headers=headers)
            
            if extract_resp.status_code == 200:
                pages_data = extract_resp.json().get("query", {}).get("pages", {})
                page_info = list(pages_data.values())[0] if pages_data else {}
                extract = page_info.get("extract", "").strip()
                
                if extract:
                    # Provide rich contextual extracts up to 600 chars
                    parsed_results.append({
                        "title": f"Wikipedia: {title}",
                        "snippet": extract[:600] + "...",
                        "link": wikipedia_article_url(title)
                    })
                    continue

            # Fallback to standard search API snippet if extracts API failed
            snippet_clean = re.sub(r'<[^>]*>', '', result.get("snippet", ""))
            parsed_results.append({
                "title": f"Wikipedia: {title}",
                "snippet": snippet_clean,
                "link": f"https://en.wikipedia.org/?curid={pageid}"
            })
        return parsed_results
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return []

# ─── LLM Connectors & Model Selection ────────────────────────────────────────

async def detect_ollama_model() -> Optional[str]:
    """
    Checks if Ollama is running and retrieves the list of installed models.
    Returns:
        Name of the first active model, or None if Ollama is offline.
    """
    client = await get_http_client()
    try:
        resp = await client.get(f"{OLLAMA_HOST}/api/tags")
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            if models_data:
                model_names = [m.get("name") for m in models_data]
                logger.info(f"Ollama running. Found models: {model_names}")
                
                # Prioritize configured default, otherwise take first available
                if OLLAMA_MODEL_DEFAULT in model_names:
                    return OLLAMA_MODEL_DEFAULT
                for name in model_names:
                    if OLLAMA_MODEL_DEFAULT in name or name.startswith(OLLAMA_MODEL_DEFAULT.split(":")[0]):
                        return name
                return model_names[0]
    except Exception as e:
        logger.warning(f"Ollama connection refused or failed at {OLLAMA_HOST}: {e}")
    return None


async def call_ollama(model_name: str, messages: List[Dict[str, str]]) -> Optional[str]:
    """Sends a chat request to the local Ollama instance."""
    client = await get_http_client()
    try:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 350
            }
        }
        logger.info(f"Calling Ollama model {model_name!r}")
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=5.0)
        if resp.status_code == 200:
            answer = resp.json().get("message", {}).get("content", "").strip()
            if answer:
                return answer
    except Exception as e:
        logger.error(f"Ollama chat generation failed: {e}")
    return None


async def call_gemini(prompt: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512}
    }
    client = await get_http_client()

    # Try each model — fail fast, max 1 retry per model, short sleep on rate-limit
    for model_name in GEMINI_MODELS:
        for attempt in range(2):  # max 2 attempts per model
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8.0)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("candidates"):
                        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    break  # 200 but no candidates, try next model

                elif resp.status_code == 429:
                    logger.warning(f"Gemini {model_name} rate limited (attempt {attempt+1}).")
                    await asyncio.sleep(1.0)  # flat 1-second wait, then try next model
                    break
                else:
                    break  # non-429 error, try next model

            except Exception as e:
                logger.error(f"Gemini error ({model_name}): {e}")
                break

    return None


async def call_llm(messages: List[Dict[str, str]], system_prompt: str) -> str:
    """
    Unified LLM router.
    Attempts Gemini API first (fast & high quality). Falls back to local Ollama if Gemini is unconfigured or rate-limited.
    """
    # 1. Try Gemini API first if GEMINI_API_KEY is configured
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        compiled_prompt_list = [f"System Instructions:\n{system_prompt}\n"]
        for msg in messages:
            role = msg["role"].capitalize()
            compiled_prompt_list.append(f"{role}: {msg['content']}")
        compiled_prompt_list.append("Assistant:")
        compiled_prompt = "\n\n".join(compiled_prompt_list)

        gemini_answer = await call_gemini(compiled_prompt)
        if gemini_answer:
            return gemini_answer
        logger.warning("Gemini call failed or rate-limited. Attempting Ollama fallback...")

    # 2. Try Ollama local model fallback
    ollama_model = await detect_ollama_model()
    if ollama_model:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        answer = await call_ollama(ollama_model, full_messages)
        if answer:
            return answer

    # 3. Critical error return
    return "Thought: I am unable to connect to Ollama or Gemini.\nFinal Answer: 🌸 I am currently having trouble connecting to my AI brains. Please verify that either Ollama is running locally, or a valid GEMINI_API_KEY is defined in your `.env` file!"

# ─── Agentic ReAct Engine ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are BloomBot, a friendly and extremely knowledgeable flower expert AI built into the BloomIQ recognition platform.
Your goal is to answer the user's SPECIFIC question directly and accurately using live web searches or Wikipedia searches.

Rules:
1. You do NOT have any internal database of flower care or gardening facts.
2. You MUST search for facts dynamically using tools if the user asks a question that needs information. Do NOT guess!
3. CRITICAL: Read the user's question carefully and answer it DIRECTLY and SPECIFICALLY.
   - If they ask "how to care for X" → explain care steps (light, water, soil, temperature).
   - If they ask "how to water X" → explain watering frequency, amount, and technique.
   - If they ask "where does X grow" → explain its native habitat, climate, regions.
   - If they ask "how to grow X" → explain planting steps, conditions, and growth tips.
   - Do NOT give a general description of the flower. Answer what was asked.
4. Do NOT mention your tool names or formatting instructions in your Final Answer.
5. You operate in a loop of Thought, Action, and Observation.
6. Always answer in English, even if search results or the user's browser locale are in another language.
7. In every turn, your response MUST be formatted in exactly one of these two ways:

Format Option A (To invoke a search tool):
Thought: [Your reasoning about what specific information is missing]
Action: [tool_name]("query")

(Note: Available tools are: `web_search` and `wikipedia_search`. The query MUST be in double quotes inside parentheses, e.g. Action: web_search("how to water orchids"))

Format Option B (To give your final answer when you have gathered all necessary information):
Thought: I have gathered all necessary information to answer the specific question.
Final Answer: [Your warm, friendly, DIRECT response that answers exactly what the user asked. Ground it strictly in the gathered search observations. Keep it concise (2-4 sentences max), and include a relevant flower emoji.]

Let's begin!"""

# Regex patterns to parse ReAct actions and final answers robustly
ACTION_REGEX = re.compile(r"Action:\s*(\w+)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE)
FINAL_ANSWER_REGEX = re.compile(r"Final\s+Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)
LLM_UNAVAILABLE_MARKERS = (
    "unable to connect to ollama or gemini",
    "trouble connecting to my ai",
)


def is_llm_unavailable_response(text: str) -> bool:
    """Detect the connector failure response returned by call_llm."""
    normalized = text.lower()
    return any(marker in normalized for marker in LLM_UNAVAILABLE_MARKERS)


# Keywords associated with each intent, used to score sentences
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "care": ["care", "caring", "maintain", "maintenance", "tips", "keep", "healthy", "light", "water", "soil", "temperature", "humidity", "fertilize", "prune"],
    "watering": ["water", "watering", "moist", "dry", "drain", "irrigat", "wet", "soak", "weekly", "daily", "thirsty"],
    "growing": ["grow", "growing", "plant", "planting", "seed", "propagat", "root", "sprout", "soil", "pot", "germinate"],
    "habitat": ["native", "habitat", "found", "region", "tropical", "grow wild", "natural", "origin", "continent", "country", "climate", "zone"],
    "sunlight": ["sun", "light", "shade", "bright", "indirect", "direct", "full sun", "partial"],
    "fertilizer": ["fertilizer", "fertilize", "feed", "nutrient", "nitrogen", "phosphorus"],
    "pruning": ["prune", "pruning", "trim", "cut", "deadhead", "shape"],
    "soil": ["soil", "pot", "repot", "mix", "drain", "pH", "compost"],
    "blooming": ["bloom", "flower", "season", "spring", "summer", "blossom"],
    "disease": ["disease", "pest", "bug", "aphid", "mold", "rot", "problem", "treat"],
}

# Sentences that are clearly just taxonomy / botanical intro — skip these
_GENERAL_INTRO_PATTERNS = [
    r"is a genus of", r"is the only genus", r"belongs to the family", r"is a species of",
    r"is a flowering plant", r"is a plant in the", r"\bgenus\b.*\bfamily\b",
    r"was first described", r"was named after", r"is named after",
]


def _is_general_intro_sentence(sentence: str) -> bool:
    """Return True if a sentence is a generic botanical classification line (not useful for care answers)."""
    s = sentence.lower()
    return any(re.search(p, s) for p in _GENERAL_INTRO_PATTERNS)


def extract_observation_sentences(
    observations: List[str],
    max_sentences: int = 4,
    intent_keywords: Optional[List[str]] = None,
) -> List[str]:
    """Pull readable fact sentences out of retrieved observation blocks.
    
    If intent_keywords are provided, sentences containing those keywords are
    ranked first so the answer matches what the user actually asked.
    """
    combined = "\n\n".join(observations)
    snippets = re.findall(r"Content:\s*(.*?)(?=\n\s*\[\d+\]\s*Title:|\Z)", combined, re.DOTALL)
    if not snippets:
        snippets = [combined]

    all_sentences: List[Tuple[int, str]] = []  # (priority, sentence)  lower = higher priority
    seen = set()

    for snippet in snippets:
        clean_snippet = re.sub(r"\s+", " ", snippet).strip()
        if clean_snippet and not clean_snippet.endswith((".", "!", "?")):
            clean_snippet += "."
        for sentence in re.split(r"(?<=[.!?])\s+", clean_snippet):
            sentence = sentence.strip()
            if not sentence or sentence.lower().startswith("no results found"):
                continue
            if not is_probably_english(sentence):
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)

            # Skip pure taxonomy/intro sentences when we have intent keywords
            if intent_keywords and _is_general_intro_sentence(sentence):
                continue

            # Score: 0 = matches intent keywords, 1 = no match
            if intent_keywords:
                s_lower = sentence.lower()
                priority = 0 if any(kw in s_lower for kw in intent_keywords) else 1
            else:
                priority = 0

            all_sentences.append((priority, sentence))

    # Sort so intent-matching sentences come first
    all_sentences.sort(key=lambda x: x[0])
    return [s for _, s in all_sentences[:max_sentences]]


def _get_intent_keywords_for_question(question: str) -> List[str]:
    """Map the user's question to a flat list of intent-matching keywords."""
    q_lower = question.lower()
    matched: List[str] = []
    for patterns, topic in QUESTION_INTENT_MAP:
        if any(p in q_lower for p in patterns):
            # Find which INTENT_KEYWORDS bucket matches this topic word
            for intent_key, kws in INTENT_KEYWORDS.items():
                if intent_key in topic.split():
                    matched.extend(kws)
                    break
            else:
                # Fallback: use the first word of the topic as a keyword
                matched.append(topic.split()[0])
    return list(set(matched)) if matched else []


def build_retrieval_fallback_answer(observations: List[str], question: str = "") -> str:
    """Create a direct, intent-aware answer from search snippets when no LLM is available."""
    intent_kws = _get_intent_keywords_for_question(question) if question else []
    facts = extract_observation_sentences(observations, max_sentences=4, intent_keywords=intent_kws or None)

    if not facts:
        facts = extract_observation_sentences(observations, max_sentences=4, intent_keywords=None)

    if facts:
        intro = "Here's what I found" if not intent_kws else "Here's what you need to know"
        return f"{intro}: {' '.join(facts)} 🌸"

    if question:
        flower_subj = extract_flower_subject(question)
        care_fb = build_care_fallback_answer(question, flower_subj)
        if care_fb:
            return care_fb

    return "🌸 I could not find specific information for that question. Please try rephrasing or ask about a different aspect of the flower."

def extract_flower_subject(question: str) -> str:
    """
    Extracts the base flower noun from the question using label mappings or fallback lists.
    Guarantees precise botanical searches on Wikipedia and DuckDuckGo.
    """
    q_lower = question.lower()
    q_lower = re.sub(r"[^a-z\s-]", " ", q_lower)

    for typo, canonical in FLOWER_ALIASES.items():
        if re.search(rf"\b{re.escape(typo)}\b", q_lower):
            return canonical
    
    # 1. Try to load from project label mappings
    flower_names = []
    try:
        import json
        mapping_path = os.path.join(os.path.dirname(__file__), "..", "data", "mappings", "label_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                raw_map = json.load(f)
                flower_names = [name.lower().strip() for name in raw_map.values()]
    except Exception as e:
        logger.error(f"Error loading flower mapping: {e}")
        
    # 2. Fallback to common flower list
    common_flowers = [
        "rose", "orchid", "sunflower", "tulip", "hibiscus", "lavender", "daisy", 
        "lily", "jasmine", "peony", "daffodil", "hydrangea", "bleeding heart", 
        "foxglove", "coneflower", "echinacea", "dandelion", "marigold", "begonia", 
        "petunia", "azalea", "bougainvillea", "carnation", "chrysanthemum", 
        "geranium", "iris", "magnolia", "pansy", "poppy", "violet", "wisteria",
        "camellia", "crocus", "freesia", "gardenia", "heather", "hyacinth",
        "lilac", "lotus", "morning glory", "primrose", "snapdragon"
    ]
    
    all_names = list(set(flower_names + common_flowers))
    # Sort by length descending to match multi-word species first
    all_names.sort(key=len, reverse=True)
    
    for flower in all_names:
        # Check singular, plural, and common modifications
        if flower in q_lower or (flower + "s") in q_lower or (flower[:-1] + "ies") in q_lower:
            return flower
            
    # 3. Last-resort heuristic fallback
    clean = q_lower
    for stopword in QUESTION_STOPWORDS:
        clean = clean.replace(stopword, "")
    clean = re.sub(r'\s+', ' ', clean).strip()

    close_match = difflib.get_close_matches(clean, all_names + list(FLOWER_ALIASES), n=1, cutoff=0.78)
    if close_match:
        return FLOWER_ALIASES.get(close_match[0], close_match[0])

    for token in clean.split():
        close_match = difflib.get_close_matches(token, all_names + list(FLOWER_ALIASES), n=1, cutoff=0.78)
        if close_match:
            return FLOWER_ALIASES.get(close_match[0], close_match[0])

    return clean


async def rag_answer(question: str) -> dict:
    """
    Executes the Agentic RAG ReAct loop.
    Optimized with Warm-Start logic: Pre-fetches the Wikipedia summary for the flower subject
    prior to starting the loop, providing immediately grounded facts in a single LLM turn to prevent 429 rate limits.
    """
    logger.info(f"Starting Agentic RAG for question: {question!r}")
    
    sources: List[Dict[str, str]] = []
    seen_links = set()
    all_observations = []
    llm_unavailable_after_retrieval = False
    
    # 1. Warm-Start: Extract clean flower subject and build a targeted, intent-aware search query
    clean_subject = extract_flower_subject(question)
    # Always pass the original question to the LLM so it knows what was asked
    normalized_question = question.strip()
    
    # Build a targeted search query based on the user's specific intent (care/watering/habitat/etc.)
    targeted_query = build_targeted_search_query(question, clean_subject) if clean_subject else question
    
    warm_start_obs = ""
    if clean_subject:
        # Use web search with the targeted query so we fetch care/watering/growing content,
        # not just general botanical facts from Wikipedia
        warm_results = await web_search(targeted_query, num_results=3)
        if not warm_results:
            # Fallback to Wikipedia if web search returns nothing
            warm_results = await wikipedia_search(clean_subject)
        if warm_results:
            formatted_snippets = []
            for i, r in enumerate(warm_results, 1):
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                formatted_snippets.append(f"[{i}] Title: {title}\nContent: {snippet}")
                if link and link not in seen_links:
                    seen_links.add(link)
                    sources.append({"title": title, "link": link})
            warm_start_obs = "\n\n".join(formatted_snippets)
            all_observations.append(warm_start_obs)

    # 2. Initialize Agent Conversation History — pass the user's original question so the LLM knows what was asked
    agent_messages: List[Dict[str, str]] = [
        {"role": "user", "content": f"Question: {normalized_question}"}
    ]
    
    if warm_start_obs:
        # Pre-populate history with the warm-start thought and observation
        agent_messages.append({
            "role": "assistant",
            "content": f"Thought: I searched for '{targeted_query}' to find specific information that answers the user's question.\nAction: web_search(\"{targeted_query}\")"
        })
        agent_messages.append({
            "role": "user",
            "content": f"Observation: {warm_start_obs}"
        })

    # 3. Agent Loop turns
    for iteration in range(MAX_ITERATIONS):
        logger.info(f"ReAct Loop Turn {iteration + 1}/{MAX_ITERATIONS}")
        
        # Query LLM (Ollama or Gemini fallback)
        llm_response = await call_llm(agent_messages, SYSTEM_PROMPT)
        logger.info(f"Agent response:\n{llm_response}")
        
        # Append LLM output to conversation trace
        agent_messages.append({"role": "assistant", "content": llm_response})
        
        # Parse for Final Answer
        final_match = FINAL_ANSWER_REGEX.search(llm_response)
        if final_match:
            final_text = final_match.group(1).strip()
            if is_llm_unavailable_response(final_text):
                logger.warning("LLM unavailable after retrieval; using source-grounded fallback answer.")
                llm_unavailable_after_retrieval = True
                break
            logger.info("Found Final Answer in ReAct loop.")
            return {
                "answer": final_text,
                "sources": sources[:4],
                "web_search_used": len(sources) > 0
            }

        # Parse for Action (if LLM wants to search further)
        action_match = ACTION_REGEX.search(llm_response)
        if action_match:
            tool_name = action_match.group(1).lower().strip()
            query_arg = action_match.group(2).strip()
            
            logger.info(f"Tool Invoked: {tool_name} with query: {query_arg!r}")
            
            tool_results = []
            if tool_name == "web_search":
                # Web search fails or is rate-limited, but we keep it active as tool
                tool_results = await web_search(query_arg, num_results=3)
            elif tool_name == "wikipedia_search":
                tool_results = await wikipedia_search(query_arg)
            else:
                observation_content = f"Error: Tool '{tool_name}' is not recognized. Use 'web_search' or 'wikipedia_search'."
                agent_messages.append({"role": "user", "content": f"Observation: {observation_content}"})
                continue
                
            # Process search results
            if tool_results:
                formatted_snippets = []
                for i, r in enumerate(tool_results, 1):
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    link = r.get("link", "")
                    
                    formatted_snippets.append(f"[{i}] Title: {title}\nContent: {snippet}")
                    
                    if link and link not in seen_links:
                        seen_links.add(link)
                        sources.append({"title": title, "link": link})
                
                observation_content = "\n\n".join(formatted_snippets)
            else:
                observation_content = f"No results found for query: '{query_arg}'"
                
            all_observations.append(observation_content)
            
            # Feed observation back to the agent
            logger.info(f"Tool executed. Observation content length: {len(observation_content)}")
            agent_messages.append({"role": "user", "content": f"Observation: {observation_content}"})
            
        else:
            logger.warning("LLM response did not fit ReAct action or final answer format.")
            # Break out of loop to run fallback generation
            break

    # ─── Resilient Fallback Direct Synthesis ─────────────────────────────────
    # If the ReAct loop exceeds iterations or fails to format correctly, we
    # synthesize a direct grounded answer using all accumulated search observations.
    
    logger.info("Executing Fallback Direct Grounded Synthesis...")

    if llm_unavailable_after_retrieval:
        fallback_results = await web_search(normalized_question, num_results=3)
        if fallback_results:
            formatted_snippets = []
            for i, r in enumerate(fallback_results, 1):
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                formatted_snippets.append(f"[{i}] Title: {title}\nContent: {snippet}")
                if link and link not in seen_links:
                    seen_links.add(link)
                    sources.append({"title": title, "link": link})
            all_observations.insert(0, "\n\n".join(formatted_snippets))
    
    # If we made no searches yet, perform one fast fallback web search
    if not all_observations:
        fallback_results = await web_search(f"{normalized_question} flower care", num_results=3)
        if fallback_results:
            formatted_snippets = []
            for i, r in enumerate(fallback_results, 1):
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                formatted_snippets.append(f"[{i}] {title}: {snippet}")
                if link and link not in seen_links:
                    seen_links.add(link)
                    sources.append({"title": title, "link": link})
            all_observations.append("\n".join(formatted_snippets))
            
    context_data = "\n\n".join(all_observations) if all_observations else "No search results available."
    
    fallback_prompt = f"""You are BloomBot, a friendly flower expert.
The user asked: "{normalized_question}"
Answer their question DIRECTLY and SPECIFICALLY using the search results provided.
- If they asked about care → explain care steps.
- If they asked about watering → explain watering schedule and technique.
- If they asked where it grows → explain native habitat and regions.
- Do NOT give a generic flower description. Answer what was actually asked.
Keep your response warm, friendly, concise (2-4 sentences max), and include a relevant flower emoji.
Ground your response strictly in the search results. If the results do not cover the question, state that you do not know.
Always answer in English.

Search Results:
{context_data}

Answer:"""

    logger.info("Calling LLM for direct synthesis.")
    # Quick direct call to call_llm (wrapping in standard user query)
    direct_messages = [{"role": "user", "content": fallback_prompt}]
    direct_response = await call_llm(direct_messages, "You are a helpful flower expert assistant.")
    
    # Strip any potential ReAct formatting in the fallback response
    direct_response_clean = re.sub(r"^(Thought|Final\s+Answer):", "", direct_response, flags=re.IGNORECASE).strip()
    if is_llm_unavailable_response(direct_response):
        # No LLM available — build a direct, intent-aware answer from the search snippets
        direct_response_clean = (
            build_care_fallback_answer(normalized_question, clean_subject)
            or build_retrieval_fallback_answer(all_observations, question=normalized_question)
        )

    return {
        "answer": direct_response_clean,
        "sources": sources[:4],
        "web_search_used": len(sources) > 0
    }
