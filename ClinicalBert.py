import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
import nltk
nltk.download('stopwords')


class BioClinicalBert:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


    def get_embeddings(self,text):
        self.text=text
        tokenized_notes = self.tokenizer(self.text, padding=True, truncation=True, return_tensors='pt', max_length=512)

        # # Obtain the embeddings for each token
        with torch.no_grad():
            self.outputs = self.model(**tokenized_notes)
        self.embeddings = self.outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # mean pooling

        return self.embeddings

    def get_smilarity_score_for_medical_concept(self):

        self.medical_concepts = pd.read_csv('MedicalConcepts.csv')
        logging.info(f"medical_concept, {self.medical_concepts}!")

        # Compute embeddings for term1 and term2
        time1 = time.time()
        self.embeddings_term1 = [self.get_embeddings(term) for term in self.medical_concepts["Term1"]]
        self.embeddings_term2 = [self.get_embeddings(term) for term in self.medical_concepts["Term2"]]

        logging.info(self.embeddings_term1)
        # Compute cosine similarity between embeddings
        similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(self.embeddings_term1, self.embeddings_term2)]
        logging.info(time.time() - time1)
        # Add similarity scores to medical concept file
        self.medical_concepts["similarity_score"] = similarities

        # Print or save the medical concept file with similarity scores
        self.medical_concepts.to_csv(r'medical_cocnept_simlarity_Score_clinicalbert.csv')
        logging.info(self.medical_concepts)

    @staticmethod
    def compute_similarity(embeddings1, embeddings2):
         return cosine_similarity([embeddings1], [embeddings2])[0][0]

    def get_clinical_notes_embedding(self):

        self.clinical_notes_df = pd.read_csv('ClinNotes.csv')
        self.clinical_notes_df['embeddings'] = self.clinical_notes_df['notes'].apply(self.get_embeddings)

        # Function to compute similarity score between two embeddings
        self.similarity_score = self.compute_similarity(self.clinical_notes_df['embeddings'][0], self.clinical_notes_df['embeddings'][1])
        logging.info(f"Similarity score between the first two clinical notes:{self.similarity_score}")

        # Save the updated DataFrame with embeddings
        self.clinical_notes_df.to_csv(r'clinical_notes_with_clinicalbert_embeddings.csv', index=False)
        self.word_embedding_visualisation(self.clinical_notes_df['embeddings'])


    def word_embedding_visualisation(self,embedding):

        self.clinical_notes_df['embeddings']=embedding
        # word visualization
        embeddings_array=np.array(self.clinical_notes_df['embeddings'].head(14).to_list())
        # Apply t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42,perplexity=4)
        embeddings_tsne = tsne.fit_transform(embeddings_array)

        # Visualize the embeddings
        plt.figure(figsize=(20,20))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
        plt.title('t-SNE Visualization of Clinical Note Embeddings')

        for i, txt in enumerate(self.clinical_notes_df['notes'].head(14)):
            # words = txt.split()
            stop_words = set(stopwords.words('english'))
            words = txt.translate(str.maketrans('', '', string.punctuation)).lower().split()
            # Split the sentence into words
            words = [word for word in words if word not in stop_words and not word.isdigit()]
            sampled_words = random.sample(words, min(len(words), 4))  # Sample up to 2 words from each sentence
            for word in sampled_words:
                # Generate random offsets for x and y positions to add space between annotations
                offset_x = random.uniform(-10, 25)
                offset_y = random.uniform(-10, 25)
                plt.annotate(word, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]), xytext=(offset_x, offset_y),
                             textcoords='offset points',fontsize=6)

        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()


def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # calling function for embedding and score
    bioclinicalbert = BioClinicalBert()
    bioclinicalbert.get_smilarity_score_for_medical_concept()
    bioclinicalbert.get_clinical_notes_embedding()


if __name__ == "__main__":
    main()
