import torch
from collections import Counter

def ensemble_max_probability(models, dataloader, device):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        max_probs = None

        for model in models:
            with torch.no_grad():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                if max_probs is None:
                    max_probs = probabilities
                else:
                    max_probs = torch.max(max_probs, probabilities)

        _, top_preds = torch.max(max_probs, dim=1)
        correct_top1 += (top_preds == labels).sum().item()

        _, top5_preds = torch.topk(max_probs, 5, dim=1)
        correct_top5 += sum([1 for i in range(len(labels)) if labels[i] in top5_preds[i]])

        total += labels.size(0)

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    return top1_accuracy, top5_accuracy



def ensemble_probability_averaging(models, dataloader, device):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        sum_probs = None

        for model in models:
            with torch.no_grad():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                if sum_probs is None:
                    sum_probs = probabilities
                else:
                    sum_probs += probabilities

        avg_probs = sum_probs / len(models)
        _, top_preds = torch.max(avg_probs, dim=1)
        correct_top1 += (top_preds == labels).sum().item()

        _, top5_preds = torch.topk(avg_probs, 5, dim=1)
        correct_top5 += sum([1 for i in range(len(labels)) if labels[i] in top5_preds[i]])

        total += labels.size(0)

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    return top1_accuracy, top5_accuracy



def custom_majority_voting(votes):
    # Assuming 'votes' is a 2D tensor of shape [batch_size, num_models]
    batch_size, num_models = votes.shape
    mode_results = torch.empty(batch_size, dtype=votes.dtype, device=votes.device)
    top5_results = torch.empty((batch_size, 5), dtype=votes.dtype, device=votes.device)
    
    for i in range(batch_size):
        vote_counts = torch.bincount(votes[i], minlength=100)  # Assuming 100 classes; adjust as needed
        top5_votes = torch.topk(vote_counts, 5).indices  # Get indices of the top 5 most frequent votes
        mode_results[i] = top5_votes[0]
        top5_results[i, :] = top5_votes
    
    return mode_results, top5_results



def ensemble_majority_voting(models, dataloader, device):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        votes = []

        for model in models:
            with torch.no_grad():
                outputs = model(inputs)
                _, top_pred = torch.max(outputs, dim=1)
                votes.append(top_pred)

        votes = torch.stack(votes).t()  # Transpose to get shape (batch_size, num_models)
        mode_results, top5_results = custom_majority_voting(votes)  # Using the custom function

        # Calculate correct predictions for Top-1
        correct_top1 += (mode_results == labels).sum().item()
        # Calculate correct predictions for Top-5
        for idx, label in enumerate(labels):
            correct_top5 += label in top5_results[idx]

        total += labels.size(0)

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    return top1_accuracy, top5_accuracy
