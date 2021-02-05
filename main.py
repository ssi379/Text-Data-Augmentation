from gluonnlp.data import SentencepieceTokenizer
import torch
import random
from embedding import PororoSentenceFactory
from kogpt2.utils import get_tokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from sentence_generate import sentence_generation

model, vocab = get_pytorch_kogpt2_model()
model.train()

se = PororoSentenceFactory(task='sentence_embedding',lang='ko',model="brainsbert.base.ko.kornli.korsts")
se_cpu = se.load('cpu')

tok_path = get_tokenizer()
tokenize = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

target = se_cpu('죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도나는 괴로와했다.')
target = torch.tensor([target]).transpose(-1, 0)


epoch = 5
loss_function = torch.nn.CosineEmbeddingLoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(epoch):

    tok = random.randint(1, 50000)
    gen_sen = sentence_generation(tok,model,vocab)
    gen_emb = se_cpu(gen_sen)
    toked = torch.tensor(vocab[tokenize(gen_sen)]).unsqueeze(0)
    data = torch.tensor([gen_emb]).transpose(-1, 0)

    optimizer.zero_grad()

    outputs = model(toked)

    loss = loss_function(data, target, )

    # loss 부분이 해결 안됨 argument 1개가 없음
    # element 0 tensor가 출력됨

    loss.backward()

    optimizer.step()

