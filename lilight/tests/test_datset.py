from .. import dataset
from .. import encoder
from .. import config

def test_dataset():
    
    c = config.Config(100,23,23,32)
    traindata = dataset.TrainData(c)
    val = next(iter(traindata.get_dataloader(2)))
    print(val)
    assert val != None
