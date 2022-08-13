import torch
from torch import nn

def train_step(optimizer, loss_fn, model, x, target):
    optimizer.zero_grad()
    y = model(x)
    loss = loss_fn(y, target)
    loss.backward()
    optimizer.step()
    return loss
     

def train(model, dataloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            x = batch["text"]
            y = batch["labels"]
            loss = train_step(optimizer, loss_fn, model, x, y)
            running_loss += loss
            if i % 10000 == 0:
                print(running_loss)
