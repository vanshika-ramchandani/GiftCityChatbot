import fitz
import faiss
import numpy as np
import re
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize
import nltk
import random

nltk.download("punkt_tab")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])

    text = re.sub(r'\b\d+\b', '', text)

    return text

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

def create_embeddings(text):
    sentences = sent_tokenize(text)
    embeddings = model(sentences).numpy()
    return sentences, embeddings

class KnowledgeBase:
    def __init__(self, text):
        self.sentences, self.embeddings = create_embeddings(text)
        self.index = faiss.IndexHNSWFlat(self.embeddings.shape[1], 32)
        self.index.add(np.array(self.embeddings))
        self.conversation_history = []
        self.previous_responses = set()

    def query(self, question, top_k=3):
        self.conversation_history.append(question)

        greetings = ["hello", "hi", "hey", "hola", "greetings"]
        greeting_responses = ["Hello! How can I assist you?", "Hi there! What can I do for you?", "Hey! Need any help?", "Hello! Ask me anything about GIFT City."]
        if question.lower() in greetings:
            return random.choice(greeting_responses)

        question_embedding = model([question]).numpy()
        distances, indices = self.index.search(question_embedding, top_k)
        retrieved_sentences = [self.sentences[i] for i in indices[0]]

        best_answer = self.select_best_answer(retrieved_sentences)

        if not best_answer.strip() or best_answer in self.previous_responses:
            best_answer = self.get_fallback_response()

        self.previous_responses.add(best_answer)
        return best_answer

    def select_best_answer(self, sentences):
        filtered_sentences = [s for s in sentences if len(s.split()) > 5]
        if filtered_sentences:
            for sentence in filtered_sentences:
                if sentence not in self.previous_responses:
                    return sentence
        return ""

    def get_fallback_response(self):
        return "I'm not sure about that. Can you ask something else related to GIFT City?"

if __name__ == "__main__":
    pdf_path = "gift.pdf"
    text = extract_text_from_pdf(pdf_path)
    kb = KnowledgeBase(text)

    print("\nChatbot: Hey there! How can I help you today?.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye! Have a great day!\n")
            break
        response = kb.query(user_input)
        print("\nChatbot:\n")
        print(response)
        print("\n" + "-" * 50 + "\n")
