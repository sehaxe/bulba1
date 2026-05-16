import pytest
import tempfile
import os
from pathlib import Path

from bulba1.tokenizer import SmartTokenizer, FastTokenizer, HFTokenizer


def test_smart_tokenizer_train(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("hello world " * 100)
        f.write("foo bar baz " * 100)

    tok = SmartTokenizer(vocab_size=50, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    assert tok.get_vocab_size() > 0
    assert tok.tokenizer is not None


def test_hf_tokenizer_train(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("test data " * 100)

    tok = HFTokenizer(vocab_size=100, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    assert tok.get_vocab_size() <= 100


def test_tokenizer_encode_decode(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("hello world " * 100)

    tok = SmartTokenizer(vocab_size=100, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    text = "hello world"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert text in decoded.lower()


def test_tokenizer_load_save(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("sample text " * 50)

    tok1 = SmartTokenizer(vocab_size=50, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok1.train([data_file])

    text = "hello world"
    ids1 = tok1.encode(text)

    tok2 = SmartTokenizer(model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok2.load()

    ids2 = tok2.encode(text)
    assert ids1 == ids2


def test_fast_tokenizer_chat_ids(temp_dir):
    import shutil
    source = "data/tokenizer_fast.json"
    dest = os.path.join(temp_dir, "tokenizer.json")
    if os.path.exists(source):
        shutil.copy(source, dest)
    else:
        data_file = os.path.join(temp_dir, "train.txt")
        with open(data_file, "w") as f:
            f.write("test " * 100)
        tok = SmartTokenizer(vocab_size=100, model_path=dest)
        tok.train([data_file])

    fast_tok = FastTokenizer(dest)
    fast_tok.load()
    chat_ids = fast_tok.chat_ids

    assert isinstance(chat_ids, dict)
    for token in FastTokenizer.CHAT_TOKENS:
        assert token in chat_ids


def test_tokenizer_batch_encode(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("hello world foo bar " * 100)

    tok = SmartTokenizer(vocab_size=100, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    texts = ["hello", "world", "foo bar"]
    ids = tok.encode_batch(texts)

    assert len(ids) == 3
    assert all(isinstance(x, list) for x in ids)


def test_tokenizer_special_tokens(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("test " * 100)

    tok = SmartTokenizer(vocab_size=50, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    pad_id = tok.tokenizer.token_to_id("<pad>")
    unk_id = tok.tokenizer.token_to_id("<unk>")

    assert pad_id is not None
    assert unk_id is not None