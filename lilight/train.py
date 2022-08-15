import torch
from torch import nn

def eval_step(model, dataloader):
    model.eval()
    guess = 0
    count = 0
    samples = 1000
    for i, batch in enumerate(dataloader):
        x = batch["text"]
        target = batch["labels"]
        y = model(x)
        v = target == torch.argmax(y, -1)
        guess += v.sum()
        count += len(v)
        if count > samples:
            break
    print("probs: ", torch.softmax(y[0], -1))
    print("guess_right : ", guess)
    print("total questions : ", count)
    print("eval_score : ", guess/count)


        #print(list(model.mlp.parameters())[0])

def train_step(optimizer, loss_fn, model, x, target):
    model.train()
    y = model(x)
    loss = loss_fn(y, target)
    
    #backprobogate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, y
     

def train(model, dataloader, lr=0.0025, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i, batch in enumerate(dataloader):
            x = batch["text"]
            y = batch["labels"]
            loss, y = train_step(optimizer, loss_fn, model, x, y)
            #print(list(model.mlp.parameters())[0])
            running_loss += loss.item()
            if i % 10000 == 0:
                print("running_Loss: ", running_loss/10000)
                print(model.seq_embedding.keys.weight)
                #print(y[0])
                #y = y.detach()
                #print("entropy: ", -1*torch.sum(y[0]*torch.log(y[0])))
                print("last_loss: ", loss.item())
                running_loss = 0.0
                print("EVAL --- ")
                eval_step(model, dataloader)
                print("------")
