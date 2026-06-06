import os
import re

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.md")

class RAGEngine:
    def __init__(self):
        self.entries = []
        self.load_kb()

    def load_kb(self):
        if not os.path.exists(KB_PATH):
            return

        with open(KB_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Split entries by "#### Entry"
        raw_entries = content.split("#### Entry")
        
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
                self.entries.append({
                    "question": question,
                    "answer": answer,
                    "tokens": self._tokenize(question),
                    "answer_tokens": self._tokenize(answer)
                })

    def _tokenize(self, text):
        # Convert to lowercase and extract words
        words = re.findall(r"\b\w+\b", text.lower())
        
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
                
        # Filter out common stop words
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", 
            "at", "by", "from", "for", "in", "out", "on", "to", "with", "is", "am", 
            "are", "was", "were", "be", "been", "being", "have", "has", "had", 
            "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "my", "your"
        }
        return [w for w in expanded_words if w not in stopwords]

    def search(self, query, threshold=0.15):
        results = self.search_multiple(query, threshold=threshold, limit=1)
        return results[0] if results else None

    def search_multiple(self, query, threshold=0.15, limit=2):
        query_tokens = self._tokenize(query)
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
