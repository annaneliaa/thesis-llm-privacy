# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


" LLM-benchmark code modified to my thesis use case "

"""
Builds the .numpy files of token sequences used for analyzing memorization, for input language

Example usage:
EUROPARL_DIR="/home/ncarlini/pile/the-eye.eu/public/AI/pile/train/"
python3 load_dataset.py $EUROPARL_DIR train en""
"""


import os
import csv
import numpy as np
import json
from transformers import GPT2Tokenizer
import sys
import hashlib

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

output_path = "datasets"
file_name = "europarl-v7.nl-en-200"
languages = ["en", "nl"]
token_length = 200

def encoder(args):
    exid, next_line, start_byte, end_byte, count = args
    encoded = tokenizer.encode(next_line, max_length=1024, truncation=True)
    sequence = encoded[start_byte:end_byte]
    return exid, sequence, count


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("USAGE: python build_pile_dataset.py $EUROPARL_DIR $SPLIT")
        exit()

    # path directory holding the dataset text files (europarl in this case)
    dataset_path = sys.argv[1]
    # split of the dataset
    split = sys.argv[2]
     # language of the dataset
    lang = sys.argv[3]

    # open the jsonlines file for the input language
    pile_files = [open(dataset_path + file_name + "." + lang + ".jsonl")]

    print("Opened file", pile_files[0].name)
    print("Generating prefix and suffix for split", split, "in language", lang)

    ds = {
        "train": "5eacff84e8b7ebdc4a364573806ddfb3", #changed this hash to my own
        "val": "318d6f3d68dfd3497956a79678067c62",
        "test": "ddcc8318f0cf1f857a6adcb18e6726b8",
        "rest": "f81027fe935c2260ae34a82e0e8c5434",
    }

    assert split in ds

    try:
        # open byte offset csv
        fin = open("datasets/" + split + "_dataset_" + lang + "_" + str(token_length) + ".csv")
    except:
        print("The split", split, "does not exist (yet?).")
        exit(1)

    prompts = {}
    counts = {}

    csvreader = csv.reader(fin)
    print("Headers:")
    print(next(csvreader))  # Skip the header row

    # Load the examples indicated by the byte offsets in the scaling dataset csv.
    for i, row in enumerate(csvreader):
        # Extract line atributes for each example from the csv
        (
            exid,
            fid,
            line_byte_offset,
            start,
            end,
            take_before,
            take_after,
            internal_offset,
            size,
            start_byte,
            end_byte,
            count,
        ) = map(int, row)

        # set file pointer to the byte offset of the line
        pile_files[fid].seek(line_byte_offset) 

        # # read the line
        # line = pile_files[fid].readline()

        # load the line
        next_line = json.loads(next(pile_files[fid]))["text"]

        tokens = tokenizer.tokenize(next_line)

        # Truncate to the specified token length
        tokens = tokens[:token_length]

        # Join the tokens back into a string
        next_line = ' '.join(tokens)

        # encode line to tokens and store in prompts dict
        if start_byte < 0:
            
            # carlini code
            next_line = bytes(next_line, "utf8")
            sequence = tokenizer.encode(
                next_line[start - take_before : end + take_after].decode(
                    "utf8", "ignore"
                  )
            )[internal_offset : internal_offset + size]
            if len(sequence) == 0:
                sequence = tokenizer.encode(
                    "z" + next_line[start : end + take_after].decode("utf8", "ignore")
                )[1 : size + 1]
        else:
            encoded = tokenizer.encode(next_line)
            sequence = encoded[start_byte:end_byte]

        if len(sequence) > 0:
            prompts[exid] = sequence
            counts[exid] = count
        else:
            print("PASS", i)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    prompts = [x[1] for x in sorted(prompts.items())]
    prompts = np.array(prompts, dtype=np.uint16)

    print("MD5 hash of dataset:")
    print(hashlib.md5(prompts.tobytes()).hexdigest())
    assert hashlib.md5(prompts.tobytes()).hexdigest() == ds[split]

    np.save(os.path.join(output_path, split + "_dataset.npy"), prompts)

  # these values should be set different for shorter token lengths (e.g. 100)
    np.save(os.path.join(output_path, split + "_preprefix.npy"), prompts[:, :100])
    np.save(os.path.join(output_path, split + "_prefix.npy"), prompts[:, 100:150])
    np.save(os.path.join(output_path, split + "_suffix.npy"), prompts[:, 150:200])

    print("Done!")
  