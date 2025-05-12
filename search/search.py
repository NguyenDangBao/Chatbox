from preprocess.preprocess import QueryCleaner
from Data.qadataset import ProductQADataset
from search.retriever import SemanticSearchQA
from sentence_transformers import SentenceTransformer
import torch
import json
import os
def main(user_query):
    # Load the dataset
    dataset = ProductQADataset("D:\Downloads\chatbox\drive-download-20250430T092955Z-001\Data\product_recommendation.jsonl")
    # Initialize the cleaner and model
    cleaner = QueryCleaner("D:\Downloads\chatbox\drive-download-20250430T092955Z-001\preprocess\english_stopwords.json")

    # Initialize the model
    model_path = os.path.abspath(r"D:\Downloads\chatbox\drive-download-20250430T092955Z-001\model\fine-tuned-miniLM-model")
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the retriever
    retriever = SemanticSearchQA(model, cleaner, dataset.get_all())

    # Test the retriever with a sample query
    result = retriever.search(user_query)
    
    print(result)
    return result