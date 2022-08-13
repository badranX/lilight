from .. import dataset
from .. import encoder

def test_dataset():
    
    c = encoder.EncoderConfig(100,23,23,32)
    loader = dataset.get_data_loader(2, c)
    val = next(iter(loader))
    print(val)
    assert val != None
