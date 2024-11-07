import random
import string
import numpy as np
from nltk.corpus import wordnet
import nltk
from transformers import AutoTokenizer, AutoModel
import torch

class RobustnessTransforms:
    """
    This class applies various transforms to text
    Allowed transforms: temperature, typos, word_swap, paraphrase, noise
    """

    def __init__(self, config):
        self.config = config
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
            
        # Initialize tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
    # Apply a robustness transform to the text
    def apply_transform(self, text, transform_type, temperature=1.0):
        if transform_type == "temperature":
            return text, temperature
        elif transform_type == "typos":
            return self.introduce_typos(text), 1.0
        elif transform_type == "word_swap":
            return self.swap_words_with_synonyms(text), 1.0
        elif transform_type == "paraphrase":
            return self.paraphrase_text(text), 1.0
        elif transform_type == "noise":
            return self.add_noise(text), 1.0
        return text, 1.0

    # Introduce typos to the text
    def introduce_typos(self, text):
        words = text.split()
        for i in range(len(words)):
            if random.random() < self.config.get("typo_probability", 0.1):
                word = words[i]
                if len(word) <= 1:
                    continue
                typo_type = random.choice(['swap', 'delete', 'insert', 'replace'])
                if typo_type == 'swap' and len(word) > 1:
                    idx = random.randint(0, len(word)-2)
                    word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
                elif typo_type == 'delete':
                    idx = random.randint(0, len(word)-1)
                    word = word[:idx] + word[idx+1:]
                elif typo_type == 'insert':
                    idx = random.randint(0, len(word))
                    word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx:]
                else:
                    idx = random.randint(0, len(word)-1)
                    word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx+1:]
                words[i] = word
        return ' '.join(words)

    # Swap words with synonyms
    def swap_words_with_synonyms(self, text):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        for i, (word, pos) in enumerate(pos_tags):
            if random.random() < self.config.get("word_swap_probability", 0.15):
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.append(lemma.name())
                if synonyms:
                    words[i] = random.choice(synonyms)
        
        return ' '.join(words)

    # Paraphrase the text
    def paraphrase_text(self, text):

        # Use embeddings to find similar sentence structure
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        
        # Add small perturbations to embeddings
        noise = torch.randn_like(embeddings) * 0.1
        perturbed_embeddings = embeddings + noise
        
        # Decode back to text
        logits = torch.matmul(perturbed_embeddings, self.model.embeddings.word_embeddings.weight.t())
        predicted_tokens = torch.argmax(logits, dim=-1)
        paraphrased = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
        
        return paraphrased[0]

    # Add noise to the text
    def add_noise(self, text):
        # Convert text to embeddings
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        
        # Add Gaussian noise
        noise = torch.randn_like(embeddings) * self.config.get("noise_std", 0.1)
        noisy_embeddings = embeddings + noise
        
        # Convert back to text
        logits = torch.matmul(noisy_embeddings, self.model.embeddings.word_embeddings.weight.t())
        predicted_tokens = torch.argmax(logits, dim=-1)
        noisy_text = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)[0]
        
        return noisy_text 