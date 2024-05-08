import csv
import multiprocessing as mp
import numpy as np
from transformers import GPT2Tokenizer

# Function to tokenize a sentence and return its length
def tokenize_sentence(sentence, tokenizer):
    tokens = tokenizer.encode(sentence, max_length=1024, truncation=True)
    return len(tokens)

# Function to generate the dataset
def generate_dataset(input_file, output_file, tokenizer):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["exid", "fid", "line_byte_offset", "start", "end", "take_before", "take_after", "internal_offset", "size", "start_byte", "end_byte", "count"])
        
        line_byte_offset = 0
        exid = 0
        fid = 0
        for line in lines:
            # Remove leading/trailing whitespaces and newline characters
            line = line.strip()
            
            # Calculate the end position (end of sentence)
            end = len(line) - 1

            # Tokenize the sentence and get its length
            size = tokenize_sentence(line, tokenizer)
            
            # Write the row to the CSV file
            csv_writer.writerow([exid, fid, line_byte_offset, 0, end, 0, 0, 0, size, -1, -1, -1])
            
            # Update line byte offset for the next sentence
            line_byte_offset += len(line) + 1  # Add 1 for the newline character
            exid += 1  # Increment example ID for the next sentence

if __name__ == "__main__":
    path = "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/nl-en/europarl-v7.nl-en.en"
    output_file = "datasets/train_dataset_en.csv"
    
    # Initialize the tokenizer (adjust this according to your tokenizer)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Generate the dataset
    generate_dataset(path, output_file, tokenizer)

    print("Dataset generated successfully!")
