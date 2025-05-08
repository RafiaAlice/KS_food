from datetime import datetime, timedelta
import os, re, json, time
from typing import List, Dict, Tuple
import numpy as np
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# --- 1. DataLoader ---
class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = self._load_data()
        self.data = self._normalize_data()
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        self.embeddings_loaded = False  # Lazy embedding flag

    def _load_data(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        with open(self.file_path) as f:
            return json.load(f)

    def _normalize_data(self) -> List[Dict]:
        normalized = []
        for entry in self.raw_data:
            county_clean = entry.get('county', '').replace(", Kansas", "").strip().title()
            norm = {
                'name': entry.get('pantry_name', 'Unknown').title().strip(),
                'address': entry.get('address', '').strip(),
                'county': county_clean,
                'city': self._extract_city(entry.get('address', '')),
                'phone': entry.get('phone', 'Not available'),
                'requirements': entry.get('tags', []),
                'hours': entry.get('weekly_hours', {}),
                'raw_hours': entry.get('raw_hours', ''),
                'link': entry.get('link', 'Not available')
            }
            normalized.append(norm)
        return normalized

    def _extract_city(self, address: str) -> str:
        parts = address.split(',')
        return parts[-2].strip().title() if len(parts) >= 2 else 'Unknown'

    def maybe_compute_embeddings(self):
        if self.embeddings_loaded:
            return
        if os.path.exists("embeddings.npy"):
            embeddings = np.load("embeddings.npy")
        else:
            embeddings = []
            for d in self.data:
                text = f"{d['name']} {d['city']} {d['county']} {d['raw_hours']} {' '.join(d['requirements'])}"
                embeddings.append(self.encoder.encode(text).astype('float32'))
            np.save("embeddings.npy", embeddings)

        for i, d in enumerate(self.data):
            d['embedding'] = embeddings[i]
        self.embeddings_loaded = True

# --- 2. IntentClassifier ---
class IntentClassifier:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=14)
        self.labels = [
            "county", "city", "zipcode", "hours", "student_only", "no_id", "tefap",
            "proof_of_residency", "mobile", "contact", "seniors", "appointment",
            "multi_day_plan", "followup"
        ]

    def detect_intents_and_entities(self, query: str) -> Tuple[List[str], Dict]:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=-1).detach().numpy()[0]
        intents = [self.labels[i] for i, prob in enumerate(probabilities) if prob > 0.5]

        entities = self._extract_entities(query)
        return intents, entities

    def _extract_entities(self, query: str) -> Dict:
        entities = {}
        zipcodes = re.findall(r"\b\d{5}\b", query)
        if zipcodes:
            entities["zipcode"] = zipcodes[0]

        pantry_names = [entry['name'] for entry in self.loader.data]
        for name in pantry_names:
            if name.lower() in query.lower():
                entities["pantry_name"] = name
                break

        return entities

# --- 3. HybridRetriever ---
class HybridRetriever:
    def __init__(self, loader: DataLoader):
        self.data = loader.data
        self.loader = loader
        self.index = None
        self.nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])
        self._build_index()

    def _build_index(self):
        self.loader.maybe_compute_embeddings()
        embed = np.vstack([d['embedding'] for d in self.data])
        quantizer = faiss.IndexFlatL2(embed.shape[1])
        self.index = faiss.IndexIVFPQ(quantizer, embed.shape[1], 16, 8)
        self.index.train(embed)
        self.index.add(embed)

    def retrieve(self, query: str, intents: List[str], entities: Dict, last_results: List[Dict] = None) -> List[Dict]:
        self.loader.maybe_compute_embeddings()
        if 'followup' in intents and 'pantry_name' in entities:
            return [p for p in last_results if p['name'] == entities['pantry_name']]

        filters = self._geo_filter(query, entities)
        q_emb = self.loader.encoder.encode(query).astype('float32')
        D, I = self.index.search(np.array([q_emb]), len(self.data))
        candidates = [self.data[i] for i in I[0]]
        out = [p for p in candidates if self._match(p, intents, filters)]
        return out[:5]

    def _geo_filter(self, query, entities):
        f = {}
        if 'county' in entities: f['county'] = entities['county']
        if 'city' in entities: f['city'] = entities['city']
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                txt = ent.text
                if 'county' in txt.lower():
                    f['county'] = txt.replace('County', '').strip().title()
                else:
                    f.setdefault('city', txt.title())
        return f

    def _match(self, p, intents, f):
        if 'county' in f and f['county'].lower() not in p['county'].lower(): return False
        if 'city' in f and f['city'].lower() not in p['city'].lower(): return False
        return True

# --- 4. ResponseGenerator ---
class ResponseGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/models/tinyllama")
        self.model = AutoModelForCausalLM.from_pretrained("/models/tinyllama")

    def generate(self, query, intents, entities, results):
        if 'followup' in intents and len(results) == 1:
            p = results[0]
            return f"""Here are the details for {p['name']}:
Address: {p['address']}
Phone: {p['phone']}
Hours: {p['raw_hours']}
Link: {p['link']}"""

        if not results:
            return "No pantries found."

        prompt = "List these food pantries in a helpful format:\n"
        for p in results:
            prompt += f"- {p['name']}, {p['address']}, {p['raw_hours']}, Phone: {p['phone']}\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 5. PantrySearchSystem ---
class PantrySearchSystem:
    def __init__(self, file_path):
        self.loader = DataLoader(file_path)
        self.intenter = IntentClassifier(self.loader)
        self.retr = None
        self.generator = ResponseGenerator()
        self.last_results = []
        self.last_interaction_time = datetime.now()

    def process(self, query: str) -> str:
        if datetime.now() - self.last_interaction_time > timedelta(minutes=15):
            self.last_results = []
            return "Your session has expired. Please start a new query."
        self.last_interaction_time = datetime.now()
        intents, entities = self.intenter.detect_intents_and_entities(query)
        results = self._get_retriever().retrieve(query, intents, entities, self.last_results)
        self.last_results = results
        return self.generator.generate(query, intents, entities, results)
