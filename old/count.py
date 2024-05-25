import nltk

def max_tokens_in_sentence(file_path):
    max_tokens = 0

    with open(file_path, 'r') as file:
        for i, line in file:
            if(i%1000) == 0:
                print(i)
            # Tokenize the sentence
            tokens = nltk.word_tokenize(line)
            num_tokens = len(tokens)
            # Update max_tokens if current sentence has more tokens
            if num_tokens > max_tokens:
                max_tokens = num_tokens
    
    return max_tokens

def main():
    path = "/Users/annavisman/stack/RUG/CS/Year3/thesis/thesis-llm-privacy/nl-en/europarl-v7.nl-en.en"
    max_tokens = max_tokens_in_sentence(path)
    print("Maximum number of tokens in a sentence:", max_tokens)

if __name__ == "__main__":
    main()