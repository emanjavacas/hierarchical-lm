
from hierarchical_lm.loader import model_loader

root = '/home/manjavacas/code/python/language-model-playground/'

path = root + 'models/RNNLanguageModel.2019-03-26+17:02:48'
m, e = model_loader(path)

seed = 'We are accounted poor citizens , the patricians'
probs, words = m.get_probabilities(e, [seed.split()])
# acts = m.get_activations(e, [seed.split()])

# path = root + 'models/CharLanguageModel.2019-03-26+10:56:35'
# m, e = model_loader(path)

# seed = 'We are accounted poor citizens, th'
# targets, probs = m.get_next_probability(e, [seed.split()])
# acts = m.get_activations(e, [seed.split()])
