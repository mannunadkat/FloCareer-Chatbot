import os
import re
import requests
import json
from dotenv import load_dotenv

# Load keys from .env file
load_dotenv()

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.md")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", 
    "at", "by", "from", "for", "in", "out", "on", "to", "with", "is", "am", 
    "are", "was", "were", "be", "been", "being", "have", "has", "had", 
    "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "my", "your",
    # Negations & Contractions to avoid positive word corrections (e.g. cant -> can)
    "cant", "dont", "wont", "shouldnt", "isnt", "didnt", "cannot", "couldnt", "wouldnt", "havent", "hasnt", "hadnt"
}

class RAGEngine:
    def __init__(self):
        self.entries = []
        self.vocabulary = set()
        
        # Detect embedding provider from environment keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if openai_key:
            self.embedding_provider = "openai"
        elif gemini_key:
            self.embedding_provider = "gemini"
        else:
            self.embedding_provider = None
            
        self.load_kb()

    def _get_openai_embedding(self, text, api_key):
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": text
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            else:
                print(f"OpenAI Embedding API error (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"Error fetching OpenAI embedding: {e}")
        return None

    def _get_gemini_embedding(self, text, api_key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()["embedding"]["values"]
            else:
                print(f"Gemini Embedding API error (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"Error fetching Gemini embedding: {e}")
        return None

    def _cosine_similarity(self, v1, v2):
        if not v1 or not v2:
            return 0.0
        dot = sum(x * y for x, y in zip(v1, v2))
        norm1 = sum(x * x for x in v1) ** 0.5
        norm2 = sum(x * x for x in v2) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def load_kb(self):
        if not os.path.exists(KB_PATH):
            return

        with open(KB_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        raw_entries = content.split("#### Entry")
        
        self.vocabulary = set()
        temp_entries = []
        
        for raw_entry in raw_entries[1:]:
            lines = raw_entry.strip().split("\n")
            if not lines:
                continue

            entry_data = {}
            current_key = None
            current_val = []

            for line in lines:
                match = re.match(r"^\*\*([^*]+)\*\*:\s*(.*)$", line.strip())
                if match:
                    if current_key:
                        entry_data[current_key] = "\n".join(current_val).strip()
                    current_key = match.group(1).strip()
                    val = match.group(2).strip()
                    current_val = [val] if val else []
                else:
                    if current_key:
                        current_val.append(line.rstrip())

            if current_key:
                entry_data[current_key] = "\n".join(current_val).strip()

            question = ""
            answer = ""

            question_keys = ["Questions", "Type of issue", "Question / Issue", "Further Assistance", "AM FAQ'S", "Unnamed: 0", "Unnamed: 1"]
            for k in question_keys:
                if k in entry_data and entry_data[k]:
                    val = str(entry_data[k]).strip()
                    if val.isdigit():
                        continue
                    if val.lower() in ["sl no", "meaning", "question / issue", "further assistance", "unnamed: 0", "unnamed: 1"]:
                        continue
                    question = val
                    break

            answer_keys = ["Resolution / Response", "Basic Trouble Shooting steps", "Action items", "Is there anything else that I can assist you with", "Response/Resolution", "Resolution", "Unnamed: 2", "Unnamed: 1"]
            for k in answer_keys:
                if k in entry_data and entry_data[k] and entry_data[k] != question:
                    val = str(entry_data[k]).strip()
                    if val.lower() in ["resolution / response", "resolution", "meaning", "unnamed: 2", "is there anything else that i can assist you with"]:
                        continue
                    answer = val
                    break

            if question and answer:
                temp_entries.append((question, answer))

        # First pass: tokenize to build vocabulary
        for question, answer in temp_entries:
            q_tokens = self._tokenize(question, is_query=False)
            a_tokens = self._tokenize(answer, is_query=False)
            self.vocabulary.update(q_tokens)
            self.vocabulary.update(a_tokens)

        # Load embeddings cache
        emb_cache = {}
        cache_updated = False
        CACHE_PATH = os.path.join(os.path.dirname(__file__), ".embeddings_cache.json")
        if self.embedding_provider:
            if os.path.exists(CACHE_PATH):
                try:
                    with open(CACHE_PATH, "r", encoding="utf-8") as cache_f:
                        emb_cache = json.load(cache_f)
                except Exception as e:
                    print(f"Error loading embeddings cache: {e}")

        # Build entries and attach embeddings
        api_key = os.environ.get("OPENAI_API_KEY") if self.embedding_provider == "openai" else os.environ.get("GEMINI_API_KEY")
        
        for question, answer in temp_entries:
            q_tokens = self._tokenize(question, is_query=False)
            a_tokens = self._tokenize(answer, is_query=False)
            
            embedding = None
            if self.embedding_provider and api_key:
                cache_key = f"{self.embedding_provider}:{question}"
                if cache_key in emb_cache:
                    embedding = emb_cache[cache_key]
                else:
                    if self.embedding_provider == "openai":
                        embedding = self._get_openai_embedding(question, api_key)
                    else:
                        embedding = self._get_gemini_embedding(question, api_key)
                    
                    if embedding:
                        emb_cache[cache_key] = embedding
                        cache_updated = True

            self.entries.append({
                "question": question,
                "answer": answer,
                "tokens": q_tokens,
                "answer_tokens": a_tokens,
                "embedding": embedding
            })

        # Save cache if updated
        if cache_updated and self.embedding_provider:
            try:
                with open(CACHE_PATH, "w", encoding="utf-8") as cache_f:
                    json.dump(emb_cache, cache_f)
            except Exception as e:
                print(f"Error saving embeddings cache: {e}")

    def _damerau_levenshtein_distance(self, s1, s2):
        d = {}
        for i in range(-1, len(s1) + 1):
            d[(i, -1)] = i + 1
        for j in range(-1, len(s2) + 1):
            d[(-1, j)] = j + 1

        for i in range(len(s1)):
            for j in range(len(s2)):
                cost = 0 if s1[i] == s2[j] else 1
                d[(i, j)] = min(
                    d[(i - 1, j)] + 1,        # deletion
                    d[(i, j - 1)] + 1,        # insertion
                    d[(i - 1, j - 1)] + cost  # substitution
                )
                if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                    d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 0.75)

        return d[(len(s1) - 1, len(s2) - 1)]

    def _find_closest_vocab_word(self, word):
        if len(word) < 3:
            return None
            
        best_word = None
        min_dist = 999
        
        for vocab_word in self.vocabulary:
            if abs(len(word) - len(vocab_word)) > 2:
                continue
            dist = self._damerau_levenshtein_distance(word, vocab_word)
            if dist < min_dist:
                min_dist = dist
                best_word = vocab_word
            elif dist == min_dist and best_word is not None:
                # Tie-breakers:
                # 1. Prefer vocab word with more character overlap (common in spelling variants)
                vocab_overlap = len(set(word).intersection(set(vocab_word)))
                best_overlap = len(set(word).intersection(set(best_word)))
                if vocab_overlap > best_overlap:
                    best_word = vocab_word
                elif vocab_overlap == best_overlap:
                    # 2. Prefer shorter word
                    if len(vocab_word) < len(best_word):
                        best_word = vocab_word
                
        max_allowed = 1 if len(word) <= 6 else 2
        if min_dist <= max_allowed:
            return best_word
        return None

    def correct_query(self, query):
        words = re.findall(r"\b\w+\b", query)
        
        corrected_words = []
        for w in words:
            w_lower = w.lower()
            if w_lower in STOPWORDS or w_lower in self.vocabulary or w_lower.isdigit():
                corrected_words.append(w)
            else:
                closest = self._find_closest_vocab_word(w_lower)
                if closest:
                    if w.isupper():
                        corrected_words.append(closest.upper())
                    elif w[0].isupper() if w else False:
                        corrected_words.append(closest.capitalize())
                    else:
                        corrected_words.append(closest)
                else:
                    corrected_words.append(w)
                    
        # Reconstruct query while keeping punctuation and spacing
        parts = re.split(r"(\b\w+\b)", query)
        new_parts = []
        word_idx = 0
        for part in parts:
            if re.match(r"^\w+$", part):
                if word_idx < len(corrected_words):
                    new_parts.append(corrected_words[word_idx])
                    word_idx += 1
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        return "".join(new_parts)

    def _tokenize(self, text, is_query=False):
        # Convert to lowercase and extract words
        words = re.findall(r"\b\w+\b", text.lower())
        
        if is_query and hasattr(self, "vocabulary") and self.vocabulary:
            corrected_words = []
            for w in words:
                if w in STOPWORDS or w in self.vocabulary or w.isdigit():
                    corrected_words.append(w)
                else:
                    closest = self._find_closest_vocab_word(w)
                    if closest:
                        corrected_words.append(closest)
                    else:
                        corrected_words.append(w)
            words = corrected_words

        # Synonym mappings for recruiting, abbreviations, and related terms
        synonyms = {
            "fb": ["feedback"],
            "feedback": ["fb"],
            "paid": ["pay", "payment"],
            "pay": ["paid", "payment"],
            "payment": ["pay", "paid"],
            "panel": ["interviewer", "evaluator"],
            "interviewer": ["panel", "evaluator"],
            "mic": ["microphone"],
            "microphone": ["mic"],
            "cam": ["camera"],
            "camera": ["cam"],
            "cand": ["candidate"],
            "candidate": ["cand"],
            "rs": ["reschedule"],
            "reschedule": ["rs"],
            "cxl": ["cancel", "cancellation"],
            "cancel": ["cxl", "cancellation"],
            "cancellation": ["cancel", "cxl"]
        }
        
        expanded_words = []
        for w in words:
            expanded_words.append(w)
            if w in synonyms:
                expanded_words.extend(synonyms[w])
                
        return [w for w in expanded_words if w not in STOPWORDS]

    def search(self, query, threshold=0.15):
        results = self.search_multiple(query, threshold=threshold, limit=1)
        return results[0] if results else None

    def search_multiple(self, query, threshold=0.15, limit=2):
        query_tokens = self._tokenize(query, is_query=True)
        if not query_tokens:
            return []

        # Generate query embedding if provider and keys are active
        query_embedding = None
        if self.embedding_provider:
            api_key = os.environ.get("OPENAI_API_KEY") if self.embedding_provider == "openai" else os.environ.get("GEMINI_API_KEY")
            if api_key:
                if self.embedding_provider == "openai":
                    query_embedding = self._get_openai_embedding(query, api_key)
                else:
                    query_embedding = self._get_gemini_embedding(query, api_key)

        scored_entries = []

        for entry in self.entries:
            entry_tokens = set(entry["tokens"]).union(set(entry["answer_tokens"]))
            intersection = set(query_tokens).intersection(entry_tokens)
            
            # If no intersection, but we have semantic search, let cosine similarity decide.
            # Otherwise, skip.
            if not intersection and not query_embedding:
                continue

            # Calculate Jaccard-like overlap score
            if intersection:
                union = set(query_tokens).union(entry_tokens)
                keyword_score = len(intersection) / len(union)
            else:
                keyword_score = 0.0

            # Boost score for exact substring match in question
            lower_q = query.lower()
            lower_entry_q = entry["question"].lower()
            if lower_q in lower_entry_q or lower_entry_q in lower_q:
                keyword_score += 0.4

            # Boost score for exact case-insensitive question match (ignoring punctuation)
            clean_q = re.sub(r"[?.,!']", "", lower_q).strip()
            clean_entry_q = re.sub(r"[?.,!']", "", lower_entry_q).strip()
            if clean_q == clean_entry_q:
                keyword_score += 1.2

            # Boost score for matches in the question specifically
            q_intersection = set(query_tokens).intersection(set(entry["tokens"]))
            keyword_score += len(q_intersection) * 0.15

            # Boost score for specific technical keyword matches
            tech_keywords = ["camera", "mic", "microphone", "audio", "voice", "volume", "editor", "join", "blank", "error", "payment", "delete", "reschedule", "cancel", "103", "104", "nivo", "ats", "video", "feed"]
            for kw in tech_keywords:
                if kw in query_tokens and kw in entry_tokens:
                    keyword_score += 0.15

            # Blend with semantic score if available
            if query_embedding and entry.get("embedding"):
                semantic_score = self._cosine_similarity(query_embedding, entry["embedding"])
                # Add semantic boost if similarity is high
                if semantic_score > 0.4:
                    score = keyword_score + (semantic_score - 0.4) * 1.5
                else:
                    score = keyword_score
            else:
                score = keyword_score

            if score >= threshold:
                scored_entries.append((score, entry))

        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_questions = set()
        for score, entry in scored_entries:
            norm_q = entry["question"].strip().lower()
            if norm_q in seen_questions:
                continue
            seen_questions.add(norm_q)
            results.append({
                "question": entry["question"],
                "answer": entry["answer"],
                "score": score
            })
            if len(results) >= limit:
                break
        return results

# Singleton instance
rag_engine = RAGEngine()
