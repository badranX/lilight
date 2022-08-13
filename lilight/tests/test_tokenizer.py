from .. import tokenizer

def test_tokenizer():
    txt = "abcd"
    out = tokenizer(txt)
    out == list(map(ord, txt))
