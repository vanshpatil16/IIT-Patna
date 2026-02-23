from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

print("Downloading sentence transformer...")
sentence_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
sentence_model.save('./models/scibert_sentence')

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer.save_pretrained('./models/scibert_tokenizer')

print("Downloading classification model...")
classification_model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", num_labels=3
)
classification_model.save_pretrained('./models/scibert_classifier')

print("Run: python -m spacy download en_core_web_lg")
