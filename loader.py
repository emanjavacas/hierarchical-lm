
import os

from model import RNNLanguageModel
from char_model import CharLanguageModel, CharLevelCorpusEncoder
from hierarchical_model import HierarchicalLanguageModel
import utils


def model_loader(modelpath):
    """
    General function to load models

    modelpath is a string /path/to/modelname (no extension)
    """

    # remove path
    modelname = os.path.basename(modelpath)

    if modelname.startswith('RNNLanguageModel'):
        model, encoder = RNNLanguageModel.load(modelpath, utils.CorpusEncoder)
    elif modelname.startswith('HierarchicalLanguageModel'):
        model, encoder = HierarchicalLanguageModel.load(modelpath, utils.CorpusEncoder)
    elif modelname.startswith('CharLanguageModel'):
        model, encoder = CharLanguageModel.load(modelpath, CharLevelCorpusEncoder)
    else:
        raise ValueError("Couldn't identify {} as model".format(modelpath))

    # always load on eval mode
    model.eval()

    return model, encoder
