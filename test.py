import torch

def test(model, dataloader, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Optionally, calculate loss and print it
            # loss = loss_fn(outputs, labels)
            # print(f'Loss: {loss.item()}')
            
    # Calculate and print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
