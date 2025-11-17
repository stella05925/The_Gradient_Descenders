# Kaggle Code submission for AMLCV
## Sapienza University of Rome
### Creators: Maria Rita Lopes, Rafal Ciolek, Stella Lin, Thomas Rames, Adam Maciejak

This competition asks you to build robust translators / adapters that map text embeddings into the target vision latent space, so that mapped text embeddings match the ground-truth image embeddings produced by the provided VAE.
You will be given a pre-trained text encoder and a pre-trained VAE (so you do not need to train those from scratch) together with a training set of examples. Your model should learn a mapping from the text-encoder space → VAE latent space that generalizes to the held-out test set.
