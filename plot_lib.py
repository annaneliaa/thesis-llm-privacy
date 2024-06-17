import wandb
import os
import json
import numpy as np
import statistics
import matplotlib.pyplot as plt

def plot_max_BLEU(exp_name, model, dataset_dir, language, example_token_len, prefix_len, num_trials):
    wandb_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_key)

    # Initialize wandb
    wandb.init(
        project="thesis-llm-privacy",
        name="Plot max. BLEU Score - " + exp_name + " - " + model,
        config={
            "experiment_name": exp_name,
            "dataset": dataset_dir,
            "language": language,
            "token_len": example_token_len,
            "prefix_len": prefix_len,
            "num_trials": num_trials,
        },
    )

    path = os.path.join("tmp", dataset_dir, language, exp_name, "bleu_scores/sorted_compl_bleu_scores.jsonl")
    print(path)
    # Load JSON data
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    print(data[0:5])

    # Extract the maximum score for each exid
    max_scores = [(int(entry['exid']), entry['scores'][0]['score']) for entry in data]

    # Sort by exid
    max_scores.sort(key=lambda x: int(x[0]))

    # Separate the exid and scores for plotting
    exids, scores = zip(*max_scores)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(exids, scores, color='b', label='Max Score', s=2)
    ax.axhline(y=0.75, color='r', linestyle='--', label='y = 0.75')

    ax.set_xlabel('exid')
    ax.set_ylabel('Max Score')
    ax.set_title('Max BLEU-score per exid for ' + exp_name + ' - ' + dataset_dir)
    ax.legend()
    ax.grid(True)

    # Log the plot to wandb
    wandb.log({"Max Scores Plot": wandb.Image(fig)})

    # Show the plot
    plt.show()

    # Finish the wandb run
    wandb.finish()

def avg_10_highest_score(exp_name, model, dataset_dir, language, example_token_len, prefix_len, num_trials, isMeteor):
    # Average of top 10 bleu scores for each exid
    wandb_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_key)

    if isMeteor:
        score_type = "meteor"
    else :
        score_type = "bleu"
    # Initialize wandb
    wandb.init(
        project="thesis-llm-privacy",
        name="Avg. of 10 highest " + score_type +  " scores - " + exp_name + " - " + model,
        config={
            "experiment_name": exp_name,
            "dataset": dataset_dir,
            "language": language,
            "token_len": example_token_len,
            "prefix_len": prefix_len,
            "num_trials": num_trials,
        },
    )

    path = os.path.join("tmp", dataset_dir, language, exp_name, "bleu_scores/sorted_compl_" + score_type + "_scores.jsonl")
    # Load JSON data
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Extract average of 10 highest scores for each exid
    max_scores = [(int(entry['exid']), statistics.mean(entry['scores'][i]['score'] for i in range(min(10, len(entry['scores']))))) for entry in data]

    # Sort by exid
    max_scores.sort(key=lambda x: int(x[0]))

    # Separate the exid and scores for plotting
    exids, scores = zip(*max_scores)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(exids, scores, color='b', label=f'Avg. {score_type} Score', s=2)
    ax.axhline(y=0.75, color='r', linestyle='--', label='y = 0.75')

    ax.set_xlabel('exid')
    ax.set_ylabel('Avg. Score')
    ax.set_title(f'Average of 10 highest {score_type} Scores per exid for ' + exp_name + ' - ' + dataset_dir)
    ax.legend()
    ax.grid(True)

    # Log the plot to wandb
    wandb.log({"Avg Scores Plot": wandb.Image(fig)})

    # Show the plot
    plt.show()

    # Finish the wandb run
    wandb.finish()

def avg_10_highest_conf(exp_name, model, dataset_dir, language, example_token_len, prefix_len, num_trials):
    # Average bleu score of 10 highest confidence generations for each exid
    wandb_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_key)

    # Initialize wandb
    wandb.init(
        project="thesis-llm-privacy",
        name="Avg. BLEU score of 10 highest confidence scores - " + exp_name + " - " + model,
        config={
            "experiment_name": exp_name,
            "dataset": dataset_dir,
            "language": language,
            "token_len": example_token_len,
            "prefix_len": prefix_len,
            "num_trials": num_trials,
        },
    )

    losses_path = os.path.join("tmp", dataset_dir, language, exp_name, "losses/decoded/sorted_compl_losses.jsonl")
    # grab first 10 highest confidence scores for each exid
    trials = []
    with open(losses_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            exid = data['exid']
            # grab last 10
            losses = data['losses'][-10:]
            trial_nums = [loss['trial'] for loss in losses]
            # trial numbers with highest confidence scores for each exid
            trials.append((exid, trial_nums))

    # Create a dictionary from trials for easy lookup
    trials_dict = {exid: trial_nums for exid, trial_nums in trials}

    score_path = os.path.join("tmp", dataset_dir, language, exp_name, "bleu_scores/sorted_compl_bleu_scores.jsonl")
    # Load JSON data
    scores = []
    with open(score_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            exid = data['exid']
            if exid in trials_dict:
                # Get the trial numbers for this exid
                trial_nums = trials_dict[exid]
                # Get the scores for these trial numbers
                trial_scores = [score for score in data['scores'] if score['trial'] in trial_nums]
                scores.append((exid, trial_scores))

    # Extract average of 10 highest scores for each exid
    max_scores = [(int(entry[0]), statistics.mean(score['score'] for score in entry[1])) for entry in scores]

    # Sort by exid
    max_scores.sort(key=lambda x: int(x[0]))

    # Extract the scores
    scores = [score for exid, score in max_scores]

    # Calculate the average score
    average_score = statistics.mean(scores)

    # Separate the exid and scores for plotting
    exids, scores = zip(*max_scores)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(exids, scores, color='b', label='Avg. BLEU Score of 10 highest confidence generations', s=2)
    ax.axhline(y=0.75, color='r', linestyle='--', label='y = 0.75')
    ax.axhline(y=average_score, color='y', linestyle='--', linewidth=2, label='Mean BLEU score')

    ax.set_xlabel('exid')
    ax.set_ylabel('Avg. Score')
    ax.set_title('Avg. BLEU Score of 10 highest confidence generations per exid for ' + exp_name + ' - ' + dataset_dir)
    ax.legend()
    ax.grid(True)

    # Log the plot to wandb
    wandb.log({"Avg Highest Confidence Scores Plot": wandb.Image(fig)})

    # Show the plot
    plt.show()

    # Finish the wandb run
    wandb.finish()