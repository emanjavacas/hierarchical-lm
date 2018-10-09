
import collections
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import torch_utils
from model import RNNLanguageModel


class CharLevelCorpusEncoder(utils.CorpusEncoder):
    def transform_batch(self, sents, conds, device='cpu'):
        if self.reverse:
            sents = [s[::-1] for s in sents]

        # word-level batch
        words, nwords = utils.CorpusEncoder.get_batch(
            [self.word.transform(s) for s in sents], self.word.pad, device)

        # char-level batch
        chars = []
        for s in sents:
            if self.reverse:    # reverse chars as well
                s = [syl[::-1] for syl in s]
            # remove syllable structure
            chars.append(self.char.transform(' '.join(s)))

        chars, nchars = utils.CorpusEncoder.get_batch(chars, self.char.pad, device)

        # conds
        bconds = {}
        for c in self.conds:
            batch = torch.tensor([self.conds[c].transform_item(d[c]) for d in conds])
            batch = batch.to(device)
            bconds[c] = batch

        return (words, nwords), (chars, nchars), bconds


class CharLanguageModel(RNNLanguageModel):
    def __init__(self, encoder, layers, emb_dim, hidden_dim, cond_dim, dropout=0.0):

        self.layers = layers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.modelname = self.get_modelname()
        super(RNNLanguageModel, self).__init__()

        cvocab = encoder.char.size()

        nll_weight = torch.ones(cvocab)
        nll_weight[encoder.char.pad] = 0
        self.register_buffer('nll_weight', nll_weight)

        # embeddings
        self.embs = nn.Embedding(cvocab, emb_dim, padding_idx=encoder.char.pad)
        input_dim = emb_dim

        # conds
        self.conds = {}
        for cond, cenc in encoder.conds.items():
            cemb = nn.Embedding(cenc.size(), cond_dim)
            self.add_module('cond_{}'.format(cond), cemb)
            self.conds[cond] = cemb
            input_dim += cond_dim

        # rnn
        rnn = []
        for layer in range(layers):
            rnn_inp = input_dim if layer == 0 else hidden_dim
            rnn_hid = hidden_dim
            rnn.append(nn.LSTM(rnn_inp, rnn_hid))
        self.rnn = nn.ModuleList(rnn)

        # output
        self.proj = nn.Linear(hidden_dim, cvocab)

        self.init()

    def init(self):
        pass

    def get_args_and_kwargs(self):
        args = self.layers, self.emb_dim, self.hidden_dim, self.cond_dim
        kwargs = {'dropout': self.dropout}
        return args, kwargs

    def forward(self, word, nwords, char, nchars, conds, hidden=None):
        # dropout!: embedding dropout (dropoute), not implemented
        # - embeddings
        # (seq x batch x emb_dim)
        embs = self.embs(char)

        # - conditions
        if conds:
            conds = [self.conds[c](conds[c]) for c in sorted(conds)]
            # expand
            seq, batch, _ = embs.size()
            conds = [c.expand(seq, batch, -1) for c in conds]
            # concatenate
            embs = torch.cat([embs, *conds], -1)

        # dropout!: input dropout (dropouti)
        embs = torch_utils.sequential_dropout(
            embs, p=self.dropout, training=self.training)

        # - rnn
        sort, unsort = torch_utils.get_sort_unsort(nchars)
        outs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], nchars[sort])
        hidden_ = []
        hidden = hidden or [None] * len(self.rnn)
        for layer, rnn in enumerate(self.rnn):
            outs, h_ = rnn(outs, hidden[layer])
            if layer != len(self.rnn) - 1:
                outs, lengths = nn.utils.rnn.pad_packed_sequence(outs)
                # dropout!: hidden dropout (dropouth)
                outs = torch_utils.sequential_dropout(
                    outs, self.dropout, self.training)
                outs = nn.utils.rnn.pack_padded_sequence(outs, lengths)
            hidden_.append(h_)
        outs, _ = nn.utils.rnn.pad_packed_sequence(outs)
        outs = outs[:, unsort]
        hidden = hidden_
        for l, h in enumerate(hidden):
            if isinstance(h, tuple):
                hidden[l] = h[0][:, unsort], h[1][:, unsort]
            else:
                hidden[l] = h[:, unsort]

        # dropout!: output dropout (dropouto)
        outs = torch_utils.sequential_dropout(outs, self.dropout, self.training)

        # logits: (seq_len x batch x vocab)
        logits = self.proj(outs)

        return logits, hidden

    def loss(self, logits, word, nwords, char, nchars):
        logits, targets = logits[:-1], char[1:]
        _, _, vocab = logits.size()

        loss = F.cross_entropy(
            logits.view(-1, vocab), targets.view(-1),
            weight=self.nll_weight, size_average=False)

        # remove 1 per batch instance
        insts = sum(nchars).item() - len(nchars)

        return loss, insts

    def loss_formatter(self, loss):
        """
        BPC for loss monitoring
        """
        return math.log2(math.e) * loss

    def sample(self, encoder, nsyms=100, batch=1,
               conds=None, hidden=None, tau=1.0, cache=None, **kwargs):
        """
        Generate stuff
        """
        device = self.device
        # batch
        if hidden is not None:
            if isinstance(hidden[0], tuple):
                batch = hidden[0][0].size(1)
            else:
                batch = hidden[0].size(1)
        else:
            hidden = [None] * len(self.rnn)

        # sample conditions if needed
        conds, bconds = conds or {}, []
        for c in sorted(self.conds):
            # sample conds
            if c not in conds:
                conds[c] = random.choice(list(encoder.conds[c].w2i.values()))
            # compute embedding
            bcond = torch.tensor([conds[c]] * batch, dtype=torch.int64).to(device)
            bcond = self.conds[c](bcond)
            bconds.append(bcond.expand(1, batch, -1))

        char = [encoder.char.bos] * batch  # (batch)
        char = torch.tensor(char, dtype=torch.int64).to(device)

        output = collections.defaultdict(list)
        mask = torch.ones(batch, dtype=torch.int64).to(device)
        scores = 0

        with torch.no_grad():
            for _ in range(nsyms):
                # check if done

                if sum(mask).item() == 0:
                    break

                # embeddings
                embs = self.embs(char.unsqueeze(0))
                if conds:
                    embs = torch.cat([embs, *bconds], -1)

                # rnn
                outs = embs
                hidden_ = []
                hidden = hidden or [None] * len(self.rnn)
                for l, rnn in enumerate(self.rnn):
                    outs, h_ = rnn(outs, hidden[l])
                    hidden_.append(h_)
                # only update hidden for active instances
                hidden = torch_utils.update_hidden(hidden, hidden_, mask)

                # get probs
                # (1 x batch x vocab) -> (batch x vocab)
                logits = self.proj(outs).squeeze(0)
                preds = F.log_softmax(logits, dim=-1)
                char = (preds / tau).exp().multinomial(1)
                score = preds.gather(1, char)
                char, score = char.squeeze(1), score.squeeze(1)

                # update mask
                mask = mask * char.ne(encoder.char.eos).long()

                # accumulate
                scores += score * mask.float()
                for idx, (active, c) in enumerate(zip(mask.tolist(), char.tolist())):
                    if active:
                        output[idx].append(encoder.char.i2w[c])

        # transform output to list-batch of hyps
        output = [output[i] for i in range(len(output))]

        # prepare output
        conds = {c: encoder.conds[c].i2w[cond] for c, cond in conds.items()}
        hyps, probs = [], []
        for hyp, score in zip(output, scores):
            try:
                prob = score.exp().item() / len(hyp)
            except ZeroDivisionError:
                prob = 0.0
            probs.append(prob)
            hyps.append(''.join(hyp[::-1] if encoder.reverse else hyp))

        return (hyps, conds), probs, hidden, cache


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--conds')
    parser.add_argument('--reverse', action='store_true',
                        help='whether to reverse input')
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--maxsize', type=int, default=10000)
    # train
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_weight', type=float, default=1.0)
    parser.add_argument('--trainer', default='Adam')
    parser.add_argument('--clipping', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--word_dropout', type=float, default=0.2)
    parser.add_argument('--minibatch', type=int, default=20)
    parser.add_argument('--repfreq', type=int, default=1000)
    parser.add_argument('--checkfreq', type=int, default=0)
    # pytorch
    parser.add_argument('--device', default='cpu')
    # extra
    parser.add_argument('--penn', action='store_true')

    args = parser.parse_args()

    from utils import LineCorpus

    print("Encoding corpus")
    start = time.time()
    conds = None
    if args.conds:
        conds = set(args.conds.split(','))
    train, dev = LineCorpus(args.train, conds=conds), LineCorpus(args.dev, conds=conds)
    encoder = CharLevelCorpusEncoder.from_corpus(
        train, dev, most_common=args.maxsize, reverse=args.reverse)
    print("... took {} secs".format(time.time() - start))

    print("Building model")
    lm = CharLanguageModel(encoder, args.layers, args.emb_dim,
                           args.hidden_dim, args.cond_dim, dropout=args.dropout)
    print(lm)
    print("Model parameters: {}".format(sum(p.nelement() for p in lm.parameters())))
    print("Storing model to path {}".format(lm.modelname))
    lm.to(args.device)

    print("Training model")
    lm.train_model(train, encoder, epochs=args.epochs, minibatch=args.minibatch,
                   dev=dev, lr=args.lr, trainer=args.trainer, clipping=args.clipping,
                   repfreq=args.repfreq, checkfreq=args.checkfreq,
                   lr_weight=args.lr_weight, bptt=args.bptt)
