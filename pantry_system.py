from datetime import datetime
import os, re, json
from typing import List, Dict, Tuple
import numpy as np
import faiss
import spacy
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. DataLoader ---
class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = self._load_data()
        self.data = self._normalize_data()
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        self.embeddings_loaded = False

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
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.labels = [
            "Find Pantry by County",
            "Find Pantry by City or Town",
            "Find Pantry by Open Hours",
            "Find Student-Only Pantry",
            "Find Pantry with No ID Required",
            "Find TEFAP Site Pantry",
            "Find Pantry Requiring Proof of Residency",
            "Find Mobile Pantry",
            "Find Pantry Contact Information",
            "Find Pantry for Seniors",
            "Find Pantry by Appointment Requirement",
            "Find Pantry by Zipcode",
            "Multi-Day Plan",
            "Follow-Up"
        ]

    def detect_intents_and_entities(self, query: str) -> Tuple[List[str], Dict]:
        prompt = f"""You are an expert assistant helping detect user intents for a food pantry search system in Kansas.

Given a user's message, identify one or more relevant intents from the following list:
{self.labels}

Return your answer as a Python list.

User: {query}
Answer:"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            intents = eval(text.split("Answer:")[-1].strip())
        except:
            intents = []

        entities = {}
        zipcodes = re.findall(r"\b\d{5}\b", query)
        if zipcodes:
            entities["zipcode"] = zipcodes[0]
        return intents, entities

# --- 3. HybridRetriever ---
class HybridRetriever:
    def __init__(self, loader: DataLoader):
        self.data = loader.data
        self.loader = loader
        self.index = None

        try:
            self.nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])
        except OSError:
            from spacy.cli import download
            print("🔄 en_core_web_sm not found. Downloading...")
            download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])

        self._build_index()

    def _build_index(self):
        self.loader.maybe_compute_embeddings()
        embed = np.vstack([d['embedding'] for d in self.data])
        self.index = faiss.IndexFlatL2(embed.shape[1])
        self.index.add(embed)

    def retrieve(self, query: str, intents: List[str], entities: Dict, last_results: List[Dict] = None) -> List[Dict]:
        self.loader.maybe_compute_embeddings()
        if 'Follow-Up' in intents and 'pantry_name' in entities:
            return [p for p in last_results if p['name'] == entities['pantry_name']]

        filters = self._geo_filter(query, entities)
        q_emb = self.loader.encoder.encode(query).astype('float32')
        D, I = self.index.search(np.array([q_emb]), len(self.data))
        candidates = [self.data[i] for i in I[0]]
        out = [p for p in candidates if self._match(p, intents, filters)]
        return out[:5]

    def _geo_filter(self, query, entities):
        f = {}
        if 'county' in entities:
            f['county'] = entities['county']
        if 'city' in entities:
            f['city'] = entities['city']

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
        if 'county' in f and f['county'].lower() not in p['county'].lower():
            return False
        if 'city' in f and f['city'].lower() not in p['city'].lower():
            return False
        return True

# --- 4. ResponseGenerator using Colab API ---
class ResponseGenerator:
    def __init__(self, colab_url: str):
        self.colab_url = colab_url

    def generate(self, query, intents, entities, results):
        payload = {
            "query": query,
            "intents": intents,
            "entities": entities,
            "results": results
        }
        try:
            response = requests.post(self.colab_url, json=payload, timeout=60)
            return response.json().get("response", "No response from Colab.")
        except Exception as e:
            print(f"Colab API error: {e}")
            return "Sorry, there was an error generating the response."

# --- 5. PantrySearchSystem ---
class PantrySearchSystem:
    def __init__(self, file_path):
        self.loader = DataLoader(file_path)
        print("🔹 Loading Flan-T5 for intent classification...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.intenter = IntentClassifier(tokenizer, model)
        self.retr = None
        self.generator = ResponseGenerator(colab_url="https://1216-34-83-84-240.ngrok-free.app/generate")
        self.last_results = []
        self.last_interaction_time = datetime.now()

    def process(self, query: str) -> str:
        print("Processing query:", query)
        intents, entities = self.intenter.detect_intents_and_entities(query)
        print("Intents:", intents)
        print("Entities:", entities)

        if self.retr is None:
            self.retr = HybridRetriever(self.loader)

        results = self.retr.retrieve(query, intents, entities, self.last_results)
        print("Retrieved results:", results)

        response = self.generator.generate(query, intents, entities, results)
        print("Final response:", response)

        self.last_results = results
        return response
