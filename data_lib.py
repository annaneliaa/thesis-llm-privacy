# imports
import csv
import json
import os
import nltk
import numpy as np


# function to count the number of examples in a dataset file with more tokens than a given threshold
# input is a CSV file with a "size" column containing the number of tokens in each example
def count_large_entries(csv_file, tokens):
    # Open the CSV file for reading
    with open(csv_file, "r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # Initialize a counter for large entries
        large_entry_count = 0

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Convert the value in the "size" column to an integer
            # this is the number of tokens in the example
            size = int(row["size"])

            # Check if the size is greater than or equal to the amount of tokens supplied
            if size >= tokens:
                # Increment the counter if the condition is met
                large_entry_count += 1

    return large_entry_count


# Function to truncate a sentence to a maximum number of tokens
def truncate_sentence(sentence, max_tokens, tokenizer):
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)

    # Truncate to max_tokens tokens
    truncated_tokens = tokens[:max_tokens]

    # Convert tokens back to string
    truncated_sentence = tokenizer.convert_tokens_to_string(truncated_tokens)

    return truncated_sentence


# Function to truncate a list of tokens to a maximum number of tokens
def truncate_tokens(tokens, max_tokens, tokenizer):
    # Truncate to max_tokens tokens
    truncated_tokens = tokens[:max_tokens]

    # Convert tokens back to string
    truncated_sentence = tokenizer.convert_tokens_to_string(truncated_tokens)

    return truncated_sentence


# Function to filter and truncate sentences in a JSONL file
def filter_truncate_json_sentences(input_file, output_file, max_tokens, tokenizer):
    print(
        "Filtering and truncating sentences in file: ",
        input_file,
        " to ",
        max_tokens,
        " tokens",
    )

    with open(input_file, "r", encoding="utf-8") as f_input, open(
        output_file, "w", encoding="utf-8"
    ) as f_output:

        for line in f_input:
            json_object = json.loads(line)

            sentence = json_object["text"]

            # Skip empty lines
            if not sentence:
                continue

            exid = json_object["exid"]
            # Remove leading/trailing whitespaces and newline characters
            sentence = sentence.strip()

            # Tokenize the sentence
            tokens = tokenizer.tokenize(sentence)

            # Check if the number of tokens exceeds the maximum
            if len(tokens) >= max_tokens:

                # Truncate the tokenized sentece to max amount of tokens
                truncated_sentence = truncate_tokens(tokens, max_tokens)

                # Create a JSON object with a "text" field containing the line
                # and the original example ID
                trunc_object = {"exid": exid, "text": truncated_sentence}

                # Write the JSON object to the output file as a single line
                json.dump(trunc_object, f_output, ensure_ascii=False)
                f_output.write("\n")


# Function to filter and truncate sentences in a text file
def filter_and_truncate_sentences(input_file, output_file, max_tokens, tokenizer):
    print(
        "Filtering and truncating sentences in file: ",
        input_file,
        " to ",
        max_tokens,
        " tokens",
    )

    with open(input_file, "r", encoding="utf-8") as f_input, open(
        output_file, "w", encoding="utf-8"
    ) as f_output:

        for line in f_input:
            # Remove leading/trailing whitespaces and newline characters
            sentence = line.strip()

            # Tokenize the sentence
            tokens = tokenizer.tokenize(sentence)

            # Check if the number of tokens exceeds the maximum
            if len(tokens) >= max_tokens:

                # Truncate the tokenized sentece to max amount of tokens
                truncated_sentence = truncate_tokens(tokens, max_tokens)

                # Write the truncated sentence to the output file
                f_output.write(truncated_sentence + "\n")


# Function to generate a csv file from the original dataset
# Columns are exid and size
# Exid is determined by the line number in the original dataset
# Inspired by Carlini code
def generate_byte_dataset(source_dir, input_file, output_file, tokenizer):
    print("Generating byte offset dataset from file: ", input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["exid", "size"])

        exid = 1  # start at 1
        for line in lines:
            # Remove leading/trailing whitespaces and newline characters
            line = line.strip()

            # Tokenize the sentence and get its length
            size = len(tokenizer.encode(line, truncation=True))

            # Write the row to the CSV file
            csv_writer.writerow([exid, size])

            exid += 1  # Always increment the example ID to keep in sync with original dataset

# This one works with JSONL files
def generate_byte_dataset_jsonl(source_dir, input_file, output_file, tokenizer):
    print("Generating byte offset dataset from file: ", input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        
    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["exid", "size"])

        for line in lines:
            # Tokenize the sentence and get its length
            json_object = json.loads(line)
            exid = json_object["exid"]
            sentence = json_object["text"]
            size = len(tokenizer.encode(sentence, truncation=True))
                
            # Write the row to the CSV file
            csv_writer.writerow([exid, size])

# Function to generate a jsonlines version of dataset
# input here is a text file
def text_to_jsonlines(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_input, \
         open(output_file, "w", encoding="utf-8") as f_output:
        id = 1

        for line in f_input:
            # Remove leading/trailing whitespaces and newline characters
            line = line.strip()

            
            # Create a JSON object with a "text" field containing the line
            json_object = {"exid": id,
                           "text": line}
            
            # Write the JSON object to the output file as a single line
            json.dump(json_object, f_output, ensure_ascii=False)
            f_output.write('\n')
            id += 1

# This function filters a CSV file based on the size column
# output file is a CSV file with the same columns as the input file
# only rows with a size greater than or equal to min_size are kept
def filter_csv(input_file, output_file, min_size):
    # Open the input CSV file for reading
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        # Create a CSV reader object
        reader = csv.DictReader(infile)
        
        # Open the output CSV file for writing
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            # Create a CSV writer object
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            
            # Write the header to the output file
            writer.writeheader()
            
            # Iterate through each row in the input file
            for row in reader:
                # Check if the size column value is at least min_size
                if int(row['size']) >= min_size:
                    # Write the row to the output file
                    writer.writerow(row)

# Function to compute a list of all exids found in an input CSV file
# Input file is a CSV with exid and size columns
def read_exids_from_csv(file):
    # integer set
    exids = set()
    with open(file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            exids.add(int(row['exid']))
    return exids, len(exids)

# Function to read a list of exids from a CSV file
# Input file is a CSV with a single column of exids
def read_common_exids(file):
    exids = []
    with open(file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            exid = row  # Strip any leading/trailing whitespace
            exid = exid[0]
            exids.append(exid)
    return exids

# Function to find the common exids between two CSV files
def find_common_exids(file1, file2):
    exids1, len1 = read_exids_from_csv(file1)
    print("Number of exids in file 1: %s", len1)
    exids2, len2 = read_exids_from_csv(file2)
    print("Number of exids in file 2: %s", len2)
    common_exids = exids1.intersection(exids2)
    # sort
    common_exids = sorted(common_exids)
    print("Number of common exids found %s", len(common_exids))
    return common_exids

# Function to write a list of exids to a CSV file
def write_exids_to_file(exids, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for exid in exids:
            writer.writerow([exid])

def trunc_json(input_file, output_file, max_tokens, exid_list, tokenizer):
    # takes common example ids from csv file and truncates the corresponding sentences in the jsonl file
    # produces a new jsonl file with the truncated sentences to length max_tokens
    print("Truncating sentences in file: ", input_file, " to ", max_tokens, " tokens")
    count = 0
    
    with open(input_file, "r", encoding="utf-8") as f_input, \
         open(output_file, "w", encoding="utf-8") as f_output:
        
        # loop over all examples in the original dataset (jsonl version)
        for line in f_input:
            # Remove leading/trailing whitespaces and newline characters
            json_object = json.loads(line)
            
            exid = json_object["exid"]
            

            if(str(exid) not in exid_list):
                continue

            else: 
                sentence = json_object["text"]
                # Tokenize the sentence
                tokens = tokenizer.tokenize(sentence)
            
                # Truncate the tokenized sentece to max amount of tokens
                truncated_sentence = truncate_tokens(tokens, max_tokens)

                trunc_obj = {"exid": exid,
                             "text": truncated_sentence}
                    
                # Write the truncated sentence to the output file
                json.dump(trunc_obj, f_output, ensure_ascii=False)
                f_output.write('\n')
                count += 1
    print("Truncated ", count, " sentences to ", output_file)
    print("Done!")

# function to concat sentences in dataset to get more examples with >= desired token length
# uses the byte offset csv file for faster processing
def process_train_data(byte_offset_csv, output_file, max_tokens):
    # Read the byte offset CSV file
    with open(byte_offset_csv, "r", newline='', encoding="utf-8") as csvfile:
        with open(output_file, "w", newline='', encoding="utf-8") as out_file:
            csv_reader = list(csv.DictReader(csvfile))
            idx = 0
            while idx < len(csv_reader):
                ids = []
                exid = int(csv_reader[idx]["exid"])
                size = int(csv_reader[idx]["size"])

                # If the current line has enough tokens, process it
                if size >= max_tokens:
                    ids.append(exid)
                    json_object = {"exid": ids, "size": size}
                    json.dump(json_object, out_file)
                    out_file.write("\n")
                    idx += 1
                else:
                    # If the next line also does not have enough tokens, concatenate them
                    if idx + 1 < len(csv_reader) and int(csv_reader[idx + 1]["size"]) < max_tokens:
                        ids.append(exid)
                        ids.append(int(csv_reader[idx + 1]["exid"]))
                        size += int(csv_reader[idx + 1]["size"])
                        json_object = {"exid": ids, "size": size}
                        json.dump(json_object, out_file)
                        out_file.write("\n")
                        idx += 2
                    else:
                        # Skip to the next line
                        idx += 1

def count_large_entries_json(json_file, max_tokens):
    # Open the JSON file for reading
    with open(json_file, "r", encoding="utf-8") as file:
        # Initialize a counter for large entries
        large_entry_count = 0
        
        # Iterate through each line in the JSON file
        for line in file:
            # Parse the JSON object from the line
            entry = json.loads(line)
            
            # Extract the size value from the JSON object
            size = entry["size"]
            
            # Check if the size is greater than or equal to the max_tokens
            if size >= max_tokens:
                # Increment the counter if the condition is met
                large_entry_count += 1
                
    return large_entry_count

def reformat_dataset(json_file, dataset_file, output_file):
    with open(json_file, "r", encoding="utf-8") as in_file, open(output_file, "w", encoding="utf-8") as outfile:
        # Initialize new exid counter
        new_exid = 1

        # Read the dataset file into a list of lines
        with open(dataset_file, 'r') as file:
            # this list will hold the dataset, starting at index 0
            dataset = file.readlines()
        
        # Get the total number of lines in the dataset
        total_lines = len(dataset)
        
        # Read the input JSON file line by line
        lines = in_file.readlines()
        for line in lines:
            # Parse the JSON object
            data = json.loads(line)
            exids = data["exid"]
            
            concat_sentence = ""
            for exid in exids:
                concat_sentence += dataset[exid-1].strip() + " "
            
            # Remove trailing space
            concat_sentence = concat_sentence.strip()
            
            new_data = {
                "exid": new_exid,
                "text": concat_sentence
            }

            new_exid += 1  # Increment the exid for the next line
            
            # Write the new JSON object to the output file
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write("\n")

# model training wants plain text, not npy so we converting concat trunc json version to plain text again
def extract_text_from_json(json_file, output_txt_file):
    with open(json_file, 'r', encoding='utf-8') as infile, open(output_txt_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            text = data["text"]
            outfile.write(text + "\n")