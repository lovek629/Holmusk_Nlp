import torch
# from transformers import AutoTokenizer, AutoModel
from transformers import BioGptTokenizer, BioGptForCausalLM
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time


class BioGPT:

    def __init__(self):
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    def get_embeddings(self,text):
        self.text=text
        tokenized_notes = self.tokenizer(self.text, padding=True, truncation=True, return_tensors='pt', max_length=512)

        # # Obtain the embeddings for each token
        with torch.no_grad():
            self.outputs = self.model(**tokenized_notes, output_hidden_states=True)
            self.embeddings = self.outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
        return self.embeddings

    def get_smilarity_score_for_medical_concept(self):

        self.medical_concepts = pd.read_csv('MedicalConcepts.csv')
        logging.info(f"medical_concept, {self.medical_concepts}!")

        # Compute embeddings for term1 and term2
        time1=time.time()
        self.embeddings_term1 = [self.get_embeddings(term) for term in self.medical_concepts["Term1"]]
        self.embeddings_term2 = [self.get_embeddings(term) for term in self.medical_concepts["Term2"]]

        logging.info(self.embeddings_term1)
        # Compute cosine similarity between embeddings
        similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(self.embeddings_term1, self.embeddings_term2)]
        logging.info(time.time()-time1)
        # Add similarity scores to medical concept file
        self.medical_concepts["similarity_score"] = similarities

        # Print or save the medical concept file with similarity scores
        self.medical_concepts.to_csv(r'medical_cocnept_simlarity_Score_BIOGPT.csv')
        print(self.medical_concepts)

    @staticmethod
    def compute_similarity(embeddings1, embeddings2):
         return cosine_similarity([embeddings1], [embeddings2])[0][0]

    def get_clinical_notes_embedding(self):

        self.clinical_notes_df = pd.read_csv('ClinNotes.csv')
        self.clinical_notes_df['embeddings'] = self.clinical_notes_df['notes'].apply(self.get_embeddings)

        # Function to compute similarity score between two embeddings
        self.similarity_score = self.compute_similarity(self.clinical_notes_df['embeddings'][0], self.clinical_notes_df['embeddings'][1])
        logging.info(f"Similarity score between the first two clinical notes:{self.similarity_score}")

        #Save the updated DataFrame with embeddings

        self.clinical_notes_df.to_csv(r'clinical_notes_with_biogpt_embeddings.csv', index=False)


def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # calling function for embedding and score
    biogpt = BioGPT()
    biogpt.get_smilarity_score_for_medical_concept()
    biogpt.get_clinical_notes_embedding()


if __name__ == "__main__":
    main()
