# vector.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  # Instead of langchain_community
from langchain_core.documents import Document
import pandas as pd

class ReviewRetriever:
    def __init__(self, csv_path="realistic_restaurant_reviews.csv"):
        # Load reviews from CSV file
        df = pd.read_csv(csv_path)
        reviews = df['Review'].tolist()
        
        # Ensure reviews are strings
        reviews = [str(review) for review in reviews]
        
        
        # Check if reviews are empty
        if not reviews:
            raise ValueError("No reviews found in the CSV file.")
        
        
        
        # Create documents from reviews
        docs = [Document(page_content=review) for review in reviews]
        
        # Initialize embeddings and vector store
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self.db = Chroma.from_documents(docs, self.embeddings, persist_directory="./chroma_db")
        self.retriever = self.db.as_retriever(search_kwargs={"k": 2})
    
    def invoke(self, query):
        # Retrieve most relevant reviews
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]

# Create an instance to export
retriever = ReviewRetriever("realistic_restaurant_reviews.csv")  # Replace with your CSV path