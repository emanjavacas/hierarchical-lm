
import json
import os
import collections
import random
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

from . import utils
from . import torch_utils
from .lstm import CustomBiLSTM


def drop_eol(word, nwords, char, nchars):
    char_mask = torch.ones_like(nchars).index_fill(0, nwords.cumsum(0) - 1, 0).byte()
    seqlen, batch = len(char), len(nchars) - word.size(1)
    char = char.masked_select(char_mask[None, :].expand_as(char)).view(seqlen, batch)
    nchars = nchars.masked_select(char_mask)
    word, nwords = word[:-1], nwords - 1
    return word, nwords, char, nchars


def prepare_input_query(encoder, sents, device, conds=None):
    (word, nwords), (char, nchars), conds = encoder.transform_batch(
        sents, conds, device)
    # drop </l> tokens
    word, nwords, char, nchars = drop_eol(word, nwords, char, nchars)
    return (word, nwords), (char, nchars), conds


class RNNLanguageModel(nn.Module):
    def __init__(self, encoder, layers, wemb_dim, cemb_dim, hidden_dim, cond_dim,
                 dropout=0.0, tie_weights=False):

        self.layers = layers
        self.wemb_dim = wemb_dim
        self.cemb_dim = cemb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.tie_weights = tie_weights
        self.modelname = self.get_modelname()
        super().__init__()

        wvocab = encoder.word.size()
        cvocab = encoder.char.size()

        nll_weight = torch.ones(wvocab)
        nll_weight[encoder.word.pad] = 0
        self.register_buffer('nll_weight', nll_weight)

        # embeddings
        self.wembs = nn.Embedding(wvocab, wemb_dim, padding_idx=encoder.word.pad)
        self.cembs = self.cembs_rnn = None
        if cemb_dim > 0:
            self.cembs = nn.Embedding(cvocab, cemb_dim, padding_idx=encoder.char.pad)
            self.cembs_rnn = CustomBiLSTM(cemb_dim, cemb_dim//2)
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
            rnn_inp = input_dim if layer == 0 else hidden_dim
            rnn_hid = hidden_dim
            if layer == layers-1 and tie_weights:
                rnn_hid = wemb_dim
            rnn.append(nn.LSTM(rnn_inp, rnn_hid))
        self.rnn = nn.ModuleList(rnn)

        # output
        if tie_weights:
            self.proj = nn.Linear(wemb_dim, wvocab)
        else:
            self.proj = nn.Linear(hidden_dim, wvocab)

        if tie_weights:
            self.proj.weight = self.wembs.weight

        self.init()

    def init(self):
        # init_range = 0.1
        # self.wembs.weight.uniform_(-init_range, init_range)
        # self.cembs.weight.uniform_(-init_range, init_range)
        # self.proj.weight.uniform_(-init_range, init_range)
        # self.proj.bias.zero_()
        pass

    def get_args_and_kwargs(self):
        args = self.layers, self.wemb_dim, self.cemb_dim, self.hidden_dim, self.cond_dim
        kwargs = {'dropout': self.dropout, 'tie_weights': self.tie_weights}
        return args, kwargs

    def save(self, dirpath, encoder):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        fpath = os.path.join(dirpath, self.modelname)

        # serialize weights
        with open(fpath + ".pt", 'wb') as f:
            device = self.device
            self.to('cpu')
            torch.save(self.state_dict(), f)
            self.to(device)

        # serialize parameters (only first time)
        if not os.path.isfile(fpath + '.params.json'):
            with open(fpath + '.params.json', 'w') as f:
                args, kwargs = self.get_args_and_kwargs()
                json.dump({'args': args, 'kwargs': kwargs}, f)

        # serialize encoder (only first time)
        if not os.path.isfile(fpath + '.encoder.json'):
            encoder.to_json(fpath + '.encoder.json')

    @classmethod
    def load(cls, modelname, encoder):
        encoder = encoder.from_json(modelname + '.encoder.json')

        with open(modelname + '.params.json') as f:
            params = json.loads(f.read())
            inst = cls(encoder, *params['args'], **params['kwargs'])
        inst.load_state_dict(torch.load(modelname + '.pt'))
        inst.modelname = os.path.basename(modelname)

        return inst, encoder

    def get_modelname(self):
        return "{}.{}".format(
            type(self).__name__, datetime.now().strftime("%Y-%m-%d+%H:%M:%S"))

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_chars(self, char, nchars, nwords):
        cembs = self.cembs(char)
        csort, cunsort = torch_utils.get_sort_unsort(nchars)
        if isinstance(self.cembs_rnn, nn.RNNBase):
            _, hidden = self.cembs_rnn(
                nn.utils.rnn.pack_padded_sequence(cembs[:, csort], nchars[csort]))
        else:
            _, hidden = self.cembs_rnn(cembs[:, csort], lengths=nchars[csort])
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        cembs = hidden[:, cunsort, :].transpose(0, 1).contiguous()
        cembs = cembs.view(sum(nwords).item(), -1)
        cembs = torch_utils.pad_flat_batch(cembs, nwords, max(nwords).item())

        return cembs

    def forward(self, word, nwords, char, nchars, conds, hidden=None, project=True):
        # dropout!: embedding dropout (dropoute), not implemented
        # (seq x batch x wemb_dim)
        embs = self.wembs(word)
        if self.cembs is not None:
            # (seq x batch x cemb_dim)
            cembs = self.embed_chars(char, nchars, nwords)
            embs = torch.cat([embs, cembs], -1)

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

        if project:
            return self.proj(outs), hidden

        return outs, hidden

    def loss(self, logits, word, nwords, char, nchars):
        logits, targets = logits[:-1], word[1:]
        _, _, vocab = logits.size()

        loss = F.cross_entropy(
            logits.view(-1, vocab), targets.view(-1),
            weight=self.nll_weight, size_average=False)

        # remove 1 per batch instance
        insts = sum(nwords).item() - len(nwords)

        return loss, insts

    def loss_formatter(self, loss):
        """
        Transform loss into a proper metric (ppl/bpc). Default to ppl.
        """
        return math.exp(min(loss, 100))

    def sample(self, encoder, nsyms=100, batch=1,
               conds=None, hidden=None, tau=1.0,
               cache=None, alpha=0.0, theta=0.0,
               avoid_unk=False):
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

        with torch.no_grad():
            for _ in range(nsyms):
                # check if done
                if sum(mask).item() == 0:
                    break

                # embeddings
                embs = self.wembs(word.unsqueeze(0))
                if self.cembs is not None:
                    cemb = self.embed_chars(char, nchars, nwords)
                    embs = torch.cat([embs, cemb], -1)
                if conds:
                    embs = torch.cat([embs, *bconds], -1)

                # rnn
                outs = embs
                hidden_ = []
                for l, rnn in enumerate(self.rnn):
                    outs, h_ = rnn(outs, hidden[l])
                    hidden_.append(h_)
                # (1 x batch x hid) -> (batch x hid)
                outs = outs.squeeze(0)
                # only update hidden for active instances
                hidden = torch_utils.update_hidden(hidden, hidden_, mask)

                # get logits
                logits = self.proj(outs)

                # - set unk to least value (might still return unk in high-entropy cases)
                if avoid_unk:
                    logits[:, encoder.word.unk] = logits.min(dim=1)[0]

                # - mix with cache
                if cache and cache.stored > 0:
                    logprob = cache.interpolate(
                        outs, logits, alpha, theta
                    ).add(1e-8).log()

                # - normal case
                else:
                    logprob = F.log_softmax(logits, dim=-1)

                # sample
                word = (logprob / tau).exp().multinomial(1)
                score = logprob.gather(1, word)
                word, score = word.squeeze(1), score.squeeze(1)

                # update mask
                mask = mask * word.ne(encoder.word.eos).long()

                # update cache if needed
                if cache:
                    cache = cache.add(outs.unsqueeze(0), word.unsqueeze(0))

                # accumulate
                scores += score * mask.float()
                for idx, (active, w) in enumerate(zip(mask.tolist(), word.tolist())):
                    if active:
                        output[idx].append(encoder.word.i2w[w])

                # get character-level input
                char = []
                for w in word.tolist():  # iterate over batch
                    w = encoder.word.i2w[w]
                    c = encoder.char.transform(w)
                    char.append(c)
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

    def dev(self, corpus, encoder, batch_size, best_loss, fails, scheduler,
            nsamples=10, target_dir='./models'):

        hidden = None
        tloss = tinsts = 0

        with torch.no_grad():
            for sents, conds in tqdm.tqdm(corpus.get_batches(batch_size)):
                (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                    sents, conds, self.device)
                logits, hidden = self(words, nwords, chars, nchars, conds, hidden)
                loss, insts = self.loss(logits, words, nwords, chars, nchars)
                tinsts += insts
                tloss += loss.item()

        tloss = self.loss_formatter(tloss / tinsts)
        print("Dev loss: {:g}".format(tloss))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(tloss)
        else:
            scheduler.step()

        if tloss < best_loss:
            print("New best dev loss: {:g}".format(tloss))
            best_loss = tloss
            fails = 0
            self.save(target_dir, encoder)
        else:
            fails += 1
            print("Failed {} time to improve best dev loss: {}".format(fails, best_loss))

        print("Sampling #{} examples".format(nsamples))
        print()
        for _ in range(nsamples):
            try:
                (hyps, conds), _, _, _ = self.sample(encoder)
                print(hyps[0], conds)  # only print first item in batch
            except Exception as e:
                print("Ooopsie while generating", str(e))
        print()

        return best_loss, fails

    def train_model(self, corpus, encoder, trainer, scheduler, epochs=5,
                    clipping=5, dev=None, minibatch=15, patience=3,
                    repfreq=1000, checkfreq=0, bptt=1, target_dir='./models'):

        # local variables
        hidden = None
        best_loss, fails = float('inf'), 0

        for e in range(epochs):

            tinsts = tloss = 0.0
            start = time.time()

            for idx, (sents, conds) in enumerate(corpus.get_batches(minibatch)):
                (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                    sents, conds, self.device)

                # early stopping
                if fails >= patience:
                    print("Early stopping after {} steps".format(fails))
                    print("Best dev loss {:g}".format(best_loss))
                    return

                # forward
                logits, hidden = self(words, nwords, chars, nchars, conds, hidden)

                # loss
                loss, insts = self.loss(logits, words, nwords, chars, nchars)
                (loss/insts).backward(retain_graph=bptt > 1)

                # bptt
                if idx % bptt == 0:
                    # step
                    nn.utils.clip_grad_norm_(self.parameters(), clipping)
                    trainer.step()
                    trainer.zero_grad()
                    # detach
                    hidden = torch_utils.detach_hidden(hidden)

                tinsts, tloss = tinsts + insts, tloss + loss.item()

                if idx and idx % (repfreq // minibatch) == 0:
                    speed = int(tinsts / (time.time() - start))
                    print("Epoch {:<3} items={:<10} loss={:<10g} items/sec={}".format(
                        e, idx * minibatch, self.loss_formatter(tloss/tinsts), speed))
                    tinsts = tloss = 0.0
                    start = time.time()

                if dev and checkfreq and idx and idx % (checkfreq // minibatch) == 0:
                    self.eval()
                    best_loss, fails = self.dev(
                        dev, encoder, minibatch, best_loss, fails, scheduler,
                        target_dir=target_dir)
                    self.train()

            if dev and not checkfreq:
                self.eval()
                best_loss, fails = self.dev(
                    dev, encoder, minibatch, best_loss, fails, scheduler,
                    target_dir=target_dir)
                self.train()

    def get_next_probability(self, encoder, sents, conds=None, hidden=None):
        (word, nwords), (char, nchars), conds = prepare_input_query(
            encoder, sents, self.device, conds=conds)
        logits, _ = self(word, nwords, char, nchars, conds, hidden=hidden)
        # get last item in sequence: (seq_len x batch x vocab) => (batch x vocab)
        probs = F.softmax(logits[-1])
        targets = [encoder.word.i2w[i] for i in range(len(encoder.word.i2w))]

        return targets, [[p.item() for p in b] for b in probs]

    def get_probabilities(self, encoder, sents, conds=None, hidden=None):
        (word, nwords), (char, nchars), conds = prepare_input_query(
            encoder, sents, self.device, conds=conds)
        logits, _ = self(word, nwords, char, nchars, conds, hidden=hidden)
        # (seq_len x batch x vocab)
        probs = F.softmax(logits, dim=2)
        # select probability assigned to true value
        # drop <l> from input and </l> tokens from probabilities
        probs = torch.gather(probs[:-1], 2, word[1:].unsqueeze(2)).squeeze(2)

        targets, word_probs = [], []
        for sent, prob in zip(sents, probs.transpose(0, 1)):
            word_probs.append(prob[:len(sent)].detach().numpy())
            targets.append(sent)

        return targets, word_probs

    def get_activations(self, encoder, sents, conds=None, hidden=None):
        # TODO: get activations from intermediate layers
        (word, nwords), (char, nchars), conds = encoder.transform_batch(
            sents, conds, self.device)
        # (seq_len x batch x hidden_dim)
        outs, _ = self(word, nwords, char, nchars, conds, hidden=hidden, project=False)
        targets, activations = [], []
        for sent, t in zip(sents, outs.transpose(0, 1)):
            t = t.detach().numpy()
            # remove <l> and </l>
            t = t[1:len(sent)+1]
            # transpose to (hidden_dim x seq_len)
            t = t.T
            targets.append(sent)
            activations.append(t)

        return targets, activations


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--conds')
    parser.add_argument('--reverse', action='store_true',
                        help='whether to reverse input')
    parser.add_argument('--wemb_dim', type=int, default=100)
    parser.add_argument('--cemb_dim', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--maxsize', type=int, default=10000)
    # train
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_weight', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1.2e-6)
    parser.add_argument('--trainer', default='Adam')
    parser.add_argument('--clipping', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--minibatch', type=int, default=20)
    parser.add_argument('--repfreq', type=int, default=1000)
    parser.add_argument('--checkfreq', type=int, default=0)
    # pytorch
    parser.add_argument('--device', default='cpu')
    # extra
    parser.add_argument('--penn', action='store_true')
    args = parser.parse_args()

    from utils import CorpusEncoder, LineCorpus

    print("Encoding corpus")
    start = time.time()
    conds = None
    if args.conds:
        conds = set(args.conds.split(','))
    train, dev = LineCorpus(args.train, conds=conds), LineCorpus(args.dev, conds=conds)
    encoder = CorpusEncoder.from_corpus(
        train, dev, most_common=args.maxsize, reverse=args.reverse)
    print("... took {} secs".format(time.time() - start))

    print("Building model")
    lm = RNNLanguageModel(encoder, args.layers, args.wemb_dim, args.cemb_dim,
                          args.hidden_dim, args.cond_dim, dropout=args.dropout,
                          tie_weights=args.tie_weights)
    print(lm)
    print("Model parameters: {}".format(sum(p.nelement() for p in lm.parameters())))
    print("Storing model to path {}".format(lm.modelname))
    lm.to(args.device)

    # trainer
    trainer = getattr(torch.optim, args.trainer)(
        lm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(trainer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, patience=1, factor=args.lr_weight)

    print("Training model")
    lm.train_model(train, encoder, trainer, scheduler,
                   epochs=args.epochs, minibatch=args.minibatch,
                   dev=dev, clipping=args.clipping, bptt=args.bptt,
                   repfreq=args.repfreq, checkfreq=args.checkfreq)

