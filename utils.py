
import json
import collections
import torch
import tqdm

# encoder
BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<l>', '</l>', '<unk>', '<pad>'


def bucket_length(length, buckets=(5, 10, 15, 20)):
    for i in sorted(buckets, reverse=True):
        if length >= i:
            return i
    return min(buckets)


class Vocab:
    def __init__(self, counter, most_common=1e+6, **reserved):
        self.w2i = {}
        self.reserved = {}
        for key, sym in reserved.items():
            if sym in counter:
                print("Removing {} [{}] from training corpus".format(key, sym))
                del counter[sym]
            self.w2i.setdefault(sym, len(self.w2i))
            self.reserved[key] = sym
            setattr(self, key, self.w2i.get(sym))

        for sym, _ in counter.most_common(int(most_common)):
            self.w2i.setdefault(sym, len(self.w2i))
        self.i2w = {i: w for w, i in self.w2i.items()}

    def size(self):
        return len(self.w2i.keys())

    def transform_item(self, item):
        try:
            return self.w2i[item]
        except KeyError:
            if self.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.unk

    def transform(self, inp):
        out = [self.transform_item(i) for i in inp]
        if self.bos is not None:
            out = [self.bos] + out
        if self.eos is not None:
            out = out + [self.eos]
        return out

    def __getitem__(self, item):
        return self.w2i[item]

    def to_dict(self):
        return {"reserved": self.reserved,
                'w2i': [{"key": key, "val": val} for key, val in self.w2i.items()]}

    @classmethod
    def from_dict(cls, d):
        inst = cls(collections.Counter())
        inst.w2i = {d["key"]: d["val"] for d in d['w2i']}
        for key, val in d['reserved'].items():
            setattr(inst, key, inst.w2i[val])
        inst.i2w = {val: key for key, val in inst.w2i.items()}

        return inst


class CorpusEncoder:
    def __init__(self, word, char, conds, reverse=False):
        self.word = word
        self.char = char
        self.conds = conds
        self.reverse = reverse

    @staticmethod
    def get_batch(sents, pad, device):
        lengths = [len(sent) for sent in sents]
        batch, maxlen = len(sents), max(lengths)
        t = torch.zeros(batch, maxlen, dtype=torch.int64) + pad
        for idx, (sent, length) in enumerate(zip(sents, lengths)):
            t[idx, :length].copy_(torch.tensor(sent))

        t = t.t().contiguous().to(device)
        lengths = torch.tensor(lengths).to(device)

        return t, lengths

    @classmethod
    def from_corpus(cls, *corpora, most_common=25000, **kwargs):
        # create counters
        w2i = collections.Counter()
        conds_w2i = collections.defaultdict(collections.Counter)
        for sent, conds, *_ in tqdm.tqdm(it for corpus in corpora for it in corpus):
            for cond in conds:
                conds_w2i[cond][conds[cond]] += 1
            for word in sent:
                w2i[word] += 1
        c2i = collections.Counter(c for w in w2i for c in w)

        # create vocabs
        word = Vocab(w2i, most_common=most_common, bos=BOS, eos=EOS, unk=UNK, pad=PAD)
        char = Vocab(c2i, eos=EOS, bos=BOS, unk=UNK, pad=PAD, eol=EOL, bol=BOL, space=' ')
        conds = {c: Vocab(cond_w2i) for c, cond_w2i in conds_w2i.items()}

        return cls(word, char, conds, **kwargs)

    def to_json(self, fpath):
        with open(fpath, 'w') as f:
            json.dump({'word': self.word.to_dict(),
                       'char': self.char.to_dict(),
                       'conds': {cond: c.to_dict() for cond, c in self.conds.items()},
                       'reverse': self.reverse},
                      f)

    @classmethod
    def from_json(cls, fpath):
        with open(fpath) as f:
            obj = json.load(f)

        word = Vocab.from_dict(obj['word'])
        char = Vocab.from_dict(obj['char'])
        conds = {cond: Vocab.from_dict(obj["conds"][cond]) for cond in obj['conds']}

        return cls(word, char, conds, obj['reverse'])

    def transform_batch(self, sents, conds, device='cpu'):  # conds is a list of dicts
        if self.reverse:
            sents = [s[::-1] for s in sents]

        # word-level batch
        words, nwords = CorpusEncoder.get_batch(
            [self.word.transform(s) for s in sents], self.word.pad, device)

        # char-level batch
        chars = []
        for sent in sents:
            sent = [self.char.transform(w) for w in sent]
            sent = [[self.char.bos, self.char.bol, self.char.eos]] + sent
            sent = sent + [[self.char.bos, self.char.eol, self.char.eos]]
            chars.extend(sent)
        chars, nchars = CorpusEncoder.get_batch(chars, self.char.pad, device)

        # conds
        bconds = {}
        for c in self.conds:
            batch = torch.tensor([self.conds[c].transform_item(d[c]) for d in conds])
            batch = batch.to(device)
            bconds[c] = batch

        return (words, nwords), (chars, nchars), bconds


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dummy_processor(line):
    return line.split(), {}


class LineCorpus:
    def __init__(self, fpath, buffer_size=100000, processor=dummy_processor, **kwargs):
        self.fpath = fpath
        self.processor = processor
        self.buffer_size = buffer_size

    def __iter__(self):
        with open(self.fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield self.processor(line)

    def prepare_buffer(self, buf, nbatches):
        buf = list(chunks(buf, nbatches))
        for batch in zip(*buf):
            sents, conds = zip(*batch)
            yield sents, conds

    def get_batches(self, batch_size):
        buf = []
        for line in iter(self):
            buf.append(line)

            if len(buf) == self.buffer_size:
                nbatches, rest = divmod(len(buf), batch_size)
                yield from self.prepare_buffer(buf[:nbatches * batch_size], nbatches)
                buf = buf[-rest:]

        if buf:
            nbatches = len(buf) // batch_size
            yield from self.prepare_buffer(buf[:nbatches * batch_size], nbatches)


def get_final_phonology(phon):
    phon = list(filter(lambda ph: ph[-1].isnumeric(), phon.split()))
    rhyme = []
    for ph in phon[::-1]:
        rhyme.append(ph)
        if ph.endswith('1'):
            break

    return rhyme[::-1]
