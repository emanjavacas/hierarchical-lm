# Conditional Hierarchical LM for text generation

This repository accompanies the paper "Generation of Hip-Hop Lyrics with Hierarchical Modeling and Conditional Templates", presented at INLG2019 in Tokyo. A link to the paper will be added as soon as it becomes public.

The repository allows you to train language models at different scales (![word](https://github.com/emanjavacas/hierarchical-lm/blob/master/hierarchical_lm/model.py), ![character](https://github.com/emanjavacas/hierarchical-lm/blob/master/hierarchical_lm/char_model.py) and ![hierarchical](https://github.com/emanjavacas/hierarchical-lm/blob/master/hierarchical_lm/hierarchical.py)) with sentence-level conditions. Additionally, the script ![generate.py](https://github.com/emanjavacas/hierarchical-lm/blob/master/hierarchical_lm/generate.py) allows you to use a pre-trained model to perform generation with optional conditions.

# Reference

If you find this code useful, please use the following reference (to be updated upon publication):

```
@inproceedings{
title = "Generation of Hip-Hop Lyrics with Hierarchical Modeling and Conditional Templates",
    author = "Manjavacas, Enrique  and
      Kestemont, Mike and
      Karsdorp, Folgert",
    booktitle = "Proceedings of the 12th International Conference on Natural Language Generation",
    month = oct,
    year = "2019",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics"
}
```
    
