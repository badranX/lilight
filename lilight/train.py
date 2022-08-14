import torch
from torch import nn

def eval_step(model, dataloader):
    for i, batch in enumerate(dataloader):
        x = batch["text"]
        y = batch["labels"]
        #print(list(model.mlp.parameters())[0])

def train_step(optimizer, loss_fn, model, x, target):
    model.train()
    y = model(x)
    loss = loss_fn(y, target)
    
    #backprobogate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
     

def train(model, dataloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i, batch in enumerate(dataloader):
            x = batch["text"]
            y = batch["labels"]
            loss = train_step(optimizer, loss_fn, model, x, y)
            #print(list(model.mlp.parameters())[0])
            running_loss += loss.item()
            if i % 10000 == 0:
                print("Loss: ", running_loss/10000)
                print(model.seq_embedding.keys.weight)
                running_loss = 0.0
