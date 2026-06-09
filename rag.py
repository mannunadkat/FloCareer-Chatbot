import os
import re

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.md")

class RAGEngine:
    def __init__(self):
        self.entries = []
        self.vocabulary = set()
        self.load_kb()

    def load_kb(self):
        if not os.path.exists(KB_PATH):
            return

        with open(KB_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Split entries by "#### Entry"
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
                # Check for standard field like **Key**: Value
                match = re.match(r"^\*\*([^*]+)\*\*:\s*(.*)$", line.strip())
                if match:
                    if current_key:
                        entry_data[current_key] = "\n".join(current_val).strip()
                    current_key = match.group(1).strip()
                    val = match.group(2).strip()
                    current_val = [val] if val else []
                else:
                    # Append multiline text
                    if current_key:
                        current_val.append(line.rstrip())

            if current_key:
                entry_data[current_key] = "\n".join(current_val).strip()

            # Now resolve what the "question" and "answer" are for this entry
            question = ""
            answer = ""

            # Check possible question fields in order of preference
            question_keys = ["Questions", "Type of issue", "Question / Issue", "Further Assistance", "AM FAQ'S", "Unnamed: 0", "Unnamed: 1"]
            for k in question_keys:
                if k in entry_data and entry_data[k]:
                    val = str(entry_data[k]).strip()
                    # Skip serial numbers / index rows
                    if val.isdigit():
                        continue
                    if val.lower() in ["sl no", "meaning", "question / issue", "further assistance", "unnamed: 0", "unnamed: 1"]:
                        continue
                    question = val
                    break

            # Check possible answer fields in order of preference
            answer_keys = ["Resolution / Response", "Basic Trouble Shooting steps", "Action items", "Is there anything else that I can assist you with", "Response/Resolution", "Resolution", "Unnamed: 2", "Unnamed: 1"]
            for k in answer_keys:
                if k in entry_data and entry_data[k] and entry_data[k] != question:
                    val = str(entry_data[k]).strip()
                    # Make sure it's not a header row
                    if val.lower() in ["resolution / response", "resolution", "meaning", "unnamed: 2", "is there anything else that i can assist you with"]:
                        continue
                    answer = val
                    break

            if question and answer:
                temp_entries.append((question, answer))

        # First pass: tokenize to build full vocabulary
        for question, answer in temp_entries:
            q_tokens = self._tokenize(question, is_query=False)
            a_tokens = self._tokenize(answer, is_query=False)
            self.vocabulary.update(q_tokens)
            self.vocabulary.update(a_tokens)
            
            self.entries.append({
                "question": question,
                "answer": answer,
                "tokens": q_tokens,
                "answer_tokens": a_tokens
            })

    def _damerau_levenshtein_distance(self, s1, s2):
        # Dynamic programming with transposition
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
                    d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 0.75) # cost of transposition is 0.75

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
                # Tie-breaker: prefer the shorter word (usually the singular root)
                if len(vocab_word) < len(best_word):
                    best_word = vocab_word
                
        max_allowed = 1 if len(word) <= 6 else 2
        if min_dist <= max_allowed:
            return best_word
        return None

    def _tokenize(self, text, is_query=False):
        # Filter out common stop words
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", 
            "at", "by", "from", "for", "in", "out", "on", "to", "with", "is", "am", 
            "are", "was", "were", "be", "been", "being", "have", "has", "had", 
            "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "my", "your"
        }

        # Convert to lowercase and extract words
        words = re.findall(r"\b\w+\b", text.lower())
        
        if is_query and hasattr(self, "vocabulary") and self.vocabulary:
            corrected_words = []
            for w in words:
                if w in stopwords or w in self.vocabulary or w.isdigit():
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
                
        return [w for w in expanded_words if w not in stopwords]

    def search(self, query, threshold=0.15):
        results = self.search_multiple(query, threshold=threshold, limit=1)
        return results[0] if results else None

    def search_multiple(self, query, threshold=0.15, limit=2):
        query_tokens = self._tokenize(query, is_query=True)
        if not query_tokens:
            return []

        scored_entries = []

        for entry in self.entries:
            entry_tokens = set(entry["tokens"]).union(set(entry["answer_tokens"]))
            intersection = set(query_tokens).intersection(entry_tokens)
            if not intersection:
                continue

            # Calculate Jaccard-like overlap score
            union = set(query_tokens).union(entry_tokens)
            score = len(intersection) / len(union)

            # Boost score for exact substring match in question
            lower_q = query.lower()
            lower_entry_q = entry["question"].lower()
            if lower_q in lower_entry_q or lower_entry_q in lower_q:
                score += 0.4

            # Boost score for exact case-insensitive question match (ignoring punctuation)
            clean_q = re.sub(r"[?.,!']", "", lower_q).strip()
            clean_entry_q = re.sub(r"[?.,!']", "", lower_entry_q).strip()
            if clean_q == clean_entry_q:
                score += 1.2

            # Boost score for matches in the question specifically
            q_intersection = set(query_tokens).intersection(set(entry["tokens"]))
            score += len(q_intersection) * 0.15

            # Boost score for specific technical keyword matches
            tech_keywords = ["camera", "mic", "microphone", "audio", "voice", "volume", "editor", "join", "blank", "error", "payment", "delete", "reschedule", "cancel", "103", "104", "nivo", "ats"]
            for kw in tech_keywords:
                if kw in query_tokens and kw in entry_tokens:
                    score += 0.15

            if score >= threshold:
                scored_entries.append((score, entry))

        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scored_entries[:limit]:
            results.append({
                "question": entry["question"],
                "answer": entry["answer"],
                "score": score
            })
        return results

# Singleton instance
rag_engine = RAGEngine()
