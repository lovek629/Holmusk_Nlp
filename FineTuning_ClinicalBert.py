import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging



# Definining a neural network for fine-tuning word embeddings

class WordEmbeddingFineTuner(nn.Module):
    def __init__(self, model,embedding_dim, vocab_size):
        super(WordEmbeddingFineTuner, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(model.get_input_embeddings().weight)
        # Freeze the parameters of the embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        logits = self.fc(embedded)
        return logits


# Defining dataset and dataloader

class ClinicalNoteDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


class Trainmodel():

    def __init__(self):
        self.model_name="emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model=AutoModel.from_pretrained(self.model_name)

        # Initialize the fine-tuning model
        self.embedding_dim = self.model.config.hidden_size  # Dimensionality of ClinicalBERT embeddings
        self.vocab_size = self.tokenizer.vocab_size  # Size of the vocabulary

        self.loaded_model = WordEmbeddingFineTuner(self.model,self.embedding_dim, self.vocab_size)


    def tokenize_and_set_model_dimension(self):

        data=pd.read_csv(r'ClinNotes.csv')
        texts= [text for text in data['notes']]
        logging.info(texts[:2])

        # Tokenize clinical note dataset
        tokenized_notes = [self.tokenizer.tokenize(text) for text in texts]

        # Convert tokens to input IDs
        input_ids = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_notes]

        dataset = ClinicalNoteDataset(input_ids)
        self.train_model_and_inference(dataset,5)


    def train_model_and_inference(self,dataset,num_epoch):
        print("no of epoch ",num_epoch)
        fine_tuner = WordEmbeddingFineTuner(self.model,self.embedding_dim, self.vocab_size)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(fine_tuner.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        # Training loop
        # num_epochs = 1
        for epoch in range(num_epoch):
            for data in dataset:
                optimizer.zero_grad()
                outputs = fine_tuner(torch.tensor(data))
                loss = criterion(outputs.view(-1, self.vocab_size), torch.tensor(data))
                loss.backward()
                optimizer.step()
                logging.info(epoch, loss)

        # Save the model
        torch.save(fine_tuner.state_dict(), 'fine_tuned_model.pth')




    def get_embeddings(self,text_to_embed):
        tokenized_text = self.tokenizer.tokenize(text_to_embed)
        # Convert tokens to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Get the embeddings using the loaded model
        with torch.no_grad():
            embeddings = self.loaded_model(torch.tensor(input_ids).unsqueeze(0))
        embeddings = embeddings.mean(dim=1).squeeze().numpy()

        return embeddings


    def get_similarity_score(self):

        # inference part start
        # Load the model
        self.loaded_model.load_state_dict(torch.load('fine_tuned_model.pth'))
        self.loaded_model.eval()

        medical_concepts= pd.read_csv('MedicalConcepts.csv')

        medical_concepts= medical_concepts.head(30)
        logging.info(medical_concepts)
        # Compute embeddings for term1 and term2
        embeddings_term1 = [self.get_embeddings(term) for term in medical_concepts["Term1"]]
        embeddings_term2 = [self.get_embeddings(term) for term in medical_concepts["Term2"]]

        # Compute cosine similarity between embeddings
        similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(embeddings_term1, embeddings_term2)]

        # Add similarity scores to medical concept file
        medical_concepts["similarity_score"] = similarities

        # Print or save the medical concept file with similarity scores
        logging.info(medical_concepts)

    def get_clinical_note_embedding(self):

        # Load clinical notes from CSV file
        clinical_notes_df = pd.read_csv('ClinNotes.csv')
        # Compute embeddings for each clinical note
        clinical_notes_df['embeddings'] = clinical_notes_df['notes'].head(10).apply(self.get_embeddings)

        # print(clinical_note_embeddings)
        # Function to compute similarity score between two embeddings
        def compute_similarity(embeddings1, embeddings2):
            return cosine_similarity([embeddings1], [embeddings2])[0][0]

        #  similarity score between the two clinical note
        similarity_score = compute_similarity(clinical_notes_df['embeddings'][0], clinical_notes_df['embeddings'][2])
        print("Similarity score between the first two clinical notes:", similarity_score)

        # Save the updated DataFrame with embeddings
        clinical_notes_df.to_csv(r'clinical_notes_with_embeddings.csv', index=False)



def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # calling function for fine tuning embedding and score
    a=Trainmodel()
    a.tokenize_and_set_model_dimension()
    a.get_similarity_score()
    a.get_clinical_note_embedding()



if __name__ == "__main__":
    main()



