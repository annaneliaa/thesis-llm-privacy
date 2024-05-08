from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

generator("Once upon a time", do_sample = True, max_length=20)