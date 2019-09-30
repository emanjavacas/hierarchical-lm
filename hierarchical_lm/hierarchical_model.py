
import math
import random
import collections
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import torch_utils
from .model import RNNLanguageModel, drop_eol
from .lstm import CustomBiLSTM


class HierarchicalLanguageModel(RNNLanguageModel):
    def __init__(self, encoder, layers, wemb_dim, cemb_dim, hidden_dim, cond_dim,
                 custom_cemb_cell=False, dropout=0.0):

        self.layers = layers
        self.wemb_dim = wemb_dim
        self.cemb_dim = cemb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.custom_cemb_cell = custom_cemb_cell
        self.modelname = self.get_modelname()
        super(RNNLanguageModel, self).__init__()

        wvocab = encoder.word.size()
        cvocab = encoder.char.size()

        nll_weight = torch.ones(cvocab)
        nll_weight[encoder.char.pad] = 0
        self.register_buffer('nll_weight', nll_weight)

        # embeddings
        self.wembs = None
        if wemb_dim > 0:
            self.wembs = nn.Embedding(wvocab, wemb_dim, padding_idx=encoder.word.pad)
        self.cembs = nn.Embedding(cvocab, cemb_dim, padding_idx=encoder.char.pad)
        if custom_cemb_cell:
            self.cembs_rnn = CustomBiLSTM(cemb_dim, cemb_dim//2)
        else:
            self.cembs_rnn = nn.LSTM(cemb_dim, cemb_dim//2, bidirectional=True)
        input_dim = wemb_dim + cemb_dim

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
            rnn.append(nn.LSTM(input_dim if layer == 0 else hidden_dim, hidden_dim))
        self.rnn = nn.ModuleList(rnn)

        # output
        self.cout_embs = nn.Embedding(cvocab, cemb_dim, padding_idx=encoder.char.pad)
        self.cout_rnn = nn.LSTM(cemb_dim + hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, cvocab)

        self.init()

    def init(self):
        for mname, m in self.named_modules():
            if isinstance(m, nn.RNNBase):
                for name, p in m.named_parameters():
                    if name.startswith('weight_ih'):
                        nn.init.orthogonal_(p)
                    elif name.startswith('weight_hh'):
                        for i in range(4):
                            nn.init.eye_(p[i*m.hidden_size: (i+1)*m.hidden_size])
                    elif name.startswith('bias'):
                        nn.init.constant_(p, 0.0)
                    else:
                        print("Unexpected parameter {} in module {}".format(name, mname))

    def get_args_and_kwargs(self):
        args = self.layers, self.wemb_dim, self.cemb_dim, self.hidden_dim, self.cond_dim
        kwargs = {'dropout': self.dropout, 'custom_cemb_cell': self.custom_cemb_cell}
        return args, kwargs

    def forward(self, word, nwords, char, nchars, conds, hidden=None, project=True):
        # - embeddings
        embs = []
        # (seq x batch x wemb_dim)
        if self.wembs is not None:
            embs.append(self.wembs(word))
        # (seq x batch x cemb_dim)
        embs.append(self.embed_chars(char, nchars, nwords))
        embs = torch.cat(embs, -1)

        # - conditions
        if conds:
            conds = [self.conds[c](conds[c]) for c in sorted(conds)]
            # expand
            seq, batch = word.size()
            conds = [c.expand(seq, batch, -1) for c in conds]
            # concatenate
            embs = torch.cat([embs, *conds], -1)

        # dropout!: input dropout (dropouti)
        embs = torch_utils.sequential_dropout(
            embs, p=self.dropout, training=self.training)

        # - rnn
        sort, unsort = torch_utils.get_sort_unsort(nwords)
        outs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], nwords[sort])
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

        # - compute char-level logits
        breaks = list(itertools.accumulate(nwords))
        # (nwords x hidden_dim)
        outs = torch_utils.flatten_padded_batch(outs, nwords)
        # indices to remove </l> from outs
        index = nwords.new([i for i in range(sum(nwords)) if i+1 not in breaks])
        # (nwords - batch x hidden_dim)
        outs = outs[index]
        # indices to remove <l> tokens from character targets
        index = nwords.new([i for i in range(sum(nwords)) if i not in breaks][1:])
        # (nchars x nwords - batch)
        char = char[:, index]
        # (nchars x nwords - batch x cemb_dim + hidden_dim)
        cemb = torch.cat([self.cout_embs(char), outs.expand(len(char), -1, -1)], -1)
        # run rnn
        chidden = None
        nchars = nchars[index]
        sort, unsort = torch_utils.get_sort_unsort(nchars)
        cemb, nchars = cemb[:, sort], nchars[sort]
        if isinstance(self.cout_rnn, nn.RNNBase):
            couts, _ = self.cout_rnn(
                nn.utils.rnn.pack_padded_sequence(cemb, nchars), chidden)
            couts, _ = nn.utils.rnn.pad_packed_sequence(couts)
        else:
            couts, _ = self.cout_rnn(cemb, chidden, nchars)
        # (nchars x nwords - batch x hidden_dim)
        couts = couts[:, unsort]
        # dropout!: char-level output dropout
        couts = torch_utils.sequential_dropout(couts, self.dropout, self.training)
        # logits: (nchars x nwords - batch x vocab)
        if project:
            return self.proj(couts), hidden

        return couts, hidden

    def loss(self, logits, word, nwords, char, nchars):
        breaks = list(itertools.accumulate(nwords))
        # indices to remove <l> tokens from targets
        index = [i for i in range(sum(nwords)) if i not in breaks][1:]
        # (nchars x nwords - batch)
        targets = char[:, torch.tensor(index).to(self.device)]

        # - remove </w> from char logits and <w> from char targets
        logits, targets = logits[:-1], targets[1:]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
            weight=self.nll_weight, size_average=False)

        # remove 1 char per word instance and 2 char per sentence (<l> tokens)
        insts = sum(nchars).item() - len(nchars) - (2 * len(nwords))

        return loss, insts

    def loss_formatter(self, loss):
        """
        BPC for loss monitoring
        """
        return math.log2(math.e) * loss

    def sample(self, encoder, nsyms=50, max_sym_len=10, batch=1,
               conds=None, hidden=None,
               tau=1.0, top_p=0, top_k=0,
               cache=None, **kwargs):
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

        word = [encoder.word.bos] * batch  # (batch)
        word = torch.tensor(word, dtype=torch.int64).to(device)
        nwords = torch.tensor([1] * batch).to(device)  # same nwords per step
        # (3 x batch)
        char = [[encoder.char.bos, encoder.char.bol, encoder.char.eos]] * batch
        char = torch.tensor(char, dtype=torch.int64).to(device).t()
        nchars = torch.tensor([3] * batch).to(device)

        output = collections.defaultdict(list)
        mask = torch.ones(batch, dtype=torch.int64).to(device)
        scores = 0
        running = torch.ones_like(mask)

        with torch.no_grad():
            for _ in range(nsyms):
                # check if done
                if sum(running).item() == 0:
                    break

                # embeddings
                embs = []
                if self.wembs is not None:
                    embs.append(self.wembs(word.unsqueeze(0)))
                embs.append(self.embed_chars(char, nchars, nwords))
                embs = torch.cat(embs, -1)
                if conds:
                    embs = torch.cat([embs, *bconds], -1)

                outs = embs
                hidden_ = []
                for l, rnn in enumerate(self.rnn):
                    outs, h_ = rnn(outs, hidden[l])
                    hidden_.append(h_)
                hidden = torch_utils.update_hidden(hidden, hidden_, mask)

                # char-level
                cinp = torch.tensor([encoder.char.bos] * batch).to(device)
                chidden = None
                coutput = collections.defaultdict(list)
                cmask = torch.ones_like(mask)

                for _ in range(max_sym_len):
                    # check if done
                    if sum(cmask).item() == 0:
                        break

                    # (1 x batch x cemb_dim + hidden_dim)
                    cemb = torch.cat([self.cout_embs(cinp.unsqueeze(0)), outs], -1)
                    # (1 x batch x hidden_dim)
                    couts, chidden = self.cout_rnn(cemb, chidden)
                    logits = self.proj(couts).squeeze(0)
                    logprob = F.log_softmax(logits, dim=-1)
                    # sample
                    cinp = torch_utils.sample_distribution(logprob, tau, top_k, top_p)
                    score = logprob.gather(1, cinp)
                    cinp, score = cinp.squeeze(1), score.squeeze(1)

                    # update char-level mask
                    cmask = cmask * cinp.ne(encoder.char.eos).long()

                    # update sentence-level mask
                    running = running * cinp.ne(encoder.char.eol).long()

                    # accumulate
                    scores += score * cmask.float()
                    for idx, (active, w) in enumerate(zip(cmask.tolist(), cinp.tolist())):
                        if active:
                            coutput[idx].append(encoder.char.i2w[w])

                # get word-level and character-level input
                word, char = [], []
                for idx, active in enumerate(running.tolist()):
                    # if not active, we still need char and word input for next step
                    w = '<dummy>'
                    if active:
                        w = ''.join(coutput.get(idx, []))  # might be empty
                        # append to global output
                        output[idx].append(w)
                    word.append(encoder.word.transform_item(w))
                    char.append(encoder.char.transform(w))

                # to batch
                word = torch.tensor(word, dtype=torch.int64).to(device)
                char, nchars = utils.CorpusEncoder.get_batch(
                    char, encoder.char.pad, device)

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
            hyps.append(' '.join(hyp[::-1] if encoder.reverse else hyp))

        return (hyps, conds), probs, hidden, cache

    def get_next_probability(self, encoder, sents, conds=None, hidden=None):
        (word, nwords), (char, nchars), conds = encoder.transform_batch(
            sents, conds, self.device)
        # (nchars x nwords - batch x vocab)
        logits, _ = self(word, nwords, char, nchars, conds, hidden=hidden)
        # project to probability
        logits = F.softmax(logits, dim=2)
        # remove </l> from nwords & nchars (</l> is already dropped in logits)
        word, nwords, char, nchars = drop_eol(word, nwords, char, nchars)
        # get last activation from logits (pre </w>)
        logits = logits[nchars-2, torch.arange(len(sents))]
        # get last word in sentence (-1 because of 0-index)
        logits = logits[nwords-1]
        return logits, word, nwords, char, nchars


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--conds')
    parser.add_argument('--reverse', action='store_true',
                        help='whether to reverse input')
    parser.add_argument('--wemb_dim', type=int, default=100)
    parser.add_argument('--cemb_dim', type=int, default=200)
    parser.add_argument('--cond_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=640)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--maxsize', type=int, default=10000)
    # train
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--lr_weight', type=float, default=0.74)
    parser.add_argument('--weight_decay', type=float, default=1.2e-6)
    parser.add_argument('--trainer', default='Adam')
    parser.add_argument('--clipping', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bptt', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--word_dropout', type=float, default=0.2)
    parser.add_argument('--minibatch', type=int, default=20)
    parser.add_argument('--repfreq', type=int, default=1000)
    parser.add_argument('--checkfreq', type=int, default=0)
    parser.add_argument('--pretrained_wembs')
    # pytorch
    parser.add_argument('--device', default='cpu')
    # extra
    parser.add_argument('--penn', action='store_true')
    parser.add_argument('--custom_cemb_cell', action='store_true')
    args = parser.parse_args()

    print("Encoding corpus")
    start = time.time()
    conds = None
    if args.conds:
        conds = set(args.conds.split(','))
    train = utils.LineCorpus(args.train, conds=conds)
    dev = utils.LineCorpus(args.dev, conds=conds)
    encoder = utils.CorpusEncoder.from_corpus(
        train, dev, most_common=args.maxsize, reverse=args.reverse)
    print("... took {} secs".format(time.time() - start))

    print("Building model")
    lm = HierarchicalLanguageModel(
        encoder, args.layers, args.wemb_dim, args.cemb_dim,
        args.hidden_dim, args.cond_dim, dropout=args.dropout,
        custom_cemb_cell=args.custom_cemb_cell)
    print(lm)
    print("Model parameters: {}".format(sum(p.nelement() for p in lm.parameters())))
    if args.pretrained_wembs:
        print("Initializing embeddings from", args.pretrained_wembs)
        torch_utils.init_pretrained_embeddings(args.pretrained_wembs, encoder, lm.wembs)
    lm.to(args.device)

    # trainer
    trainer = getattr(torch.optim, args.trainer)(
        lm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(trainer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer,
        list(range(0, 20, 5)) + list(range(20, 30, 5)) + list(range(30, args.epochs, 2)),
        gamma=args.lr_weight)

    print("Training model")
    print("Storing model to path {}".format(lm.modelname))
    lm.train_model(train, encoder, trainer, scheduler,
                   epochs=args.epochs, minibatch=args.minibatch,
                   dev=dev, clipping=args.clipping, bptt=args.bptt,
                   repfreq=args.repfreq, checkfreq=args.checkfreq)
