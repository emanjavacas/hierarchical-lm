

import json
from gensim.summarization import keywords


def load_verses(path='data/ohhla-new.jsonl'):
    with open(path, errors='ignore') as f:
        for idx, line in enumerate(f):
            try:
                for verse in json.loads(line)['text']:
                    yield [w['token'] for line in verse for w in line if w['token']]
            except json.decoder.JSONDecodeError:
                print("Couldn't read song #{}".format(idx+1))


verses = load_verses()

for _ in range(100):
    try:
        text = ' '.join(next(verses))
        ks = keywords(text, ratio=0.25).split('\n')
        print(text)
        print()
        print("Keywords: ", '-'.join(ks))
        print()
        print('-' * 10)
        print()
    except:
        pass
