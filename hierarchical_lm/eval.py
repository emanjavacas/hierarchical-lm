
import tqdm
import torch

from loader import model_loader
from utils import LineCorpus


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('corpus_path')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    model, encoder = model_loader(args.path)
    model.eval()
    model.to(args.device)
    print(model)
    print("* n params", sum(p.nelement() for p in model.parameters()))

    corpus = LineCorpus(args.corpus_path)

    hidden = None
    tloss = tinsts = 0
    with torch.no_grad():
        for sents, conds in tqdm.tqdm(corpus.get_batches(1)):
            (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                sents, conds, args.device)
            logits, hidden = model(words, nwords, chars, nchars, conds, hidden)
            loss, insts = model.loss(logits, words, nwords, chars, nchars)
            tinsts += insts
            tloss += loss.item()

    tloss = model.loss_formatter(tloss / tinsts)
    print("Loss: {:g}".format(tloss))

