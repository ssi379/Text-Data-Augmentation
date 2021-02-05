from gluonnlp.data import SentencepieceTokenizer
import torch
from kogpt2.utils import get_tokenizer

def sentence_generation(random_tok,model_,vocab_):
    tok_path = get_tokenizer()
    tok = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)
    n=0
    sent=''
    while n<30:
        if n==0:
            input_ids = torch.tensor([vocab_[vocab_.bos_token],]  + [random_tok]).unsqueeze(0)
        else:
            input_ids = torch.tensor([vocab_[vocab_.bos_token], ] + vocab_[toked]).unsqueeze(0)
        pred = model_(input_ids)[0]
        gen = vocab_.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
        if gen == '</s>':
            break
        sent += gen.replace('▁', ' ')
        toked = tok(sent)
        n+=1

    return sent
