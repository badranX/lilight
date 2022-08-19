import torch
from .config import *
from torch import nn
from .dataset import Dataset
from .encoder import Encoder


def eval_step(model, dataloader, samples=None):
    model.eval()
    guess = 0
    count = 0
    for i, batch in enumerate(dataloader):
        x = batch["text"]
        target = batch["labels"]
        y = model(x)
        v = target == torch.argmax(y, -1)
        guess += v.sum()
        count += len(v)
        if samples and samples > count:
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
     

def train(model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    losses = []
    log_i = 1000
    count = 0.
    for epoch in range(config.epochs):
        print("Epoch: ", epoch)
        for i, batch in enumerate(config.train_dataloader):
            x = batch["text"]
            y = batch["labels"]
            loss, y = train_step(optimizer, loss_fn, model, x, y)
            #print(list(model.mlp.parameters())[0])
            running_loss += loss.item()
            count += 1
            if i % log_i == 0:
                print("i: ", i)
                print("running_Loss: ", running_loss/count)
                running_loss = 0.0
                count = 0.
                print("last_loss: ", loss.item())
        #end for
        print("EVAL --- ")
        eval_step(model, config.eval_dataloader, samples= 5000)
        print("------")
    #end for
    device = torch.device('cpu')
    model.to(device)
    torch.save(model, "model.pt")



def start_train():
    config = Config()
    traindata = Dataset(config)
    train_dataloader = traindata.get_dataloader(batch_size = config.class_count, split="train")
    eval_dataloader = traindata.get_dataloader(batch_size = config.class_count, split="validation")
    config.train_dataloader = train_dataloader
    config.eval_dataloader = train_dataloader
    model = Encoder(config)
    train(model, config)


if __name__ == "__main__":
    start_train()
