import torch
def evaluate(model, images, labels):
    model.eval()

    correct= 0
    total = 0

    with torch.no_grad():
        for img, label in zip(images, labels):
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

            if label == "class_a":
                target = 0
            else:
                target = 1

            outputs = model(img)
            predicted= torch.argmax(outputs, dim=1).item()

            if predicted == target:
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}