import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


def save_model(model, filename):
    # Save the model's parameters to a file
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename):
    # Load the model's parameters from a file 
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return model



def plot_loss(losses, val_losses, epoch, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Matplotlib Plot
    plt.figure()
    epochs = range(1, epoch + 2)
    plt.plot(epochs, losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.grid(True)
    
    # Save the matplotlib plot in the specified directory
    matplotlib_filename = os.path.join(directory, f"matplotlib_loss_plot_epoch_{epoch+1}.png")
    plt.savefig(matplotlib_filename)
    plt.close()
    print(f"Matplotlib loss plot saved as {matplotlib_filename}")

    # Plotly Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(epochs), y=losses, mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=list(epochs), y=val_losses, mode='lines+markers', name='Validation Loss'))
    fig.update_layout(
        title='Training and Validation Loss Over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark'
    )

    # Save the plotly plot in the specified directory
    plotly_filename = os.path.join(directory, f"plotly_loss_plot_epoch_{epoch+1}.html")
    fig.write_html(plotly_filename)
    print(f"Plotly loss plot saved as {plotly_filename}")



def evaluate_model(model, data_loader, device):
    #evaluate the model on the given data loader and return top-1 and top-5 accuracy
    model.to(device)
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            
            # Top-1 accuracy
            correct_top1 += (preds[:, 0] == labels).sum().item()
            
            # Top-5 accuracy
            for i in range(labels.size(0)):
                if labels[i] in preds[i]:
                    correct_top5 += 1

            total += labels.size(0)
    
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    print(f"Top-1 Accuracy: {top1_accuracy}%")
    print(f"Top-5 Accuracy: {top5_accuracy}%")
    
    return top1_accuracy, top5_accuracy

def get_device():
    # Return the avail device 
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
