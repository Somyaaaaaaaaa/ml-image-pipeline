import torch
import torch.nn as nn
import torch.optim as optim

def train(model, images, labels, epochs = 5):
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0

        for img, label in zip(images, labels):
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

            if label == "class_a":
                target = torch.tensor([0])

            else:
                target = torch.tensor([1])

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model
