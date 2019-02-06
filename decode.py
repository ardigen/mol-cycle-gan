import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm

from jtvae import (Vocab,
                   JTNNVAE)


class Options:
    def __init__(self,
                 jtvae_path="./jtvae/",
                 hidden_size=450,
                 latent_size=56,
                 depth=3,
                 jtnn_model_path="molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4",
                 vocab_path="data/zinc/vocab.txt"):
        self.jtvae_path = jtvae_path
        self.vocab_path = os.path.join(jtvae_path, vocab_path)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.model_path = os.path.join(jtvae_path, jtnn_model_path)


def load_model(opts):
    vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depth = int(opts.depth)

    model = JTNNVAE(vocab, hidden_size, latent_size, depth)
    model.load_state_dict(torch.load(opts.model_path))

    return model.cuda()


def decode_from_jtvae(data_path, opts, model):
    smiles_df = pd.read_csv(data_path, index_col=0)
    mols = smiles_df.values
    returned_smiles = []

    tree_dims = int(opts.latent_size / 2)

    for i in tqdm.tqdm(range(mols.shape[0])):
        tree_vec = np.expand_dims(mols[i, 0:tree_dims], 0)
        mol_vec = np.expand_dims(mols[i, tree_dims:], 0)
        tree_vec = torch.autograd.Variable(torch.from_numpy(tree_vec).cuda().float())
        mol_vec = torch.autograd.Variable(torch.from_numpy(mol_vec).cuda().float())
        smi = model.decode(tree_vec, mol_vec, prob_decode=False)
        returned_smiles.append(smi)

    return returned_smiles


def decode(jtvae_path_tuple,
           jtvae_setting_tuple,
           encoding_data_tuple):
    jtvae_path, jtnn_model_path, vocab_path = jtvae_path_tuple
    hidden_size, latent_size, depth = jtvae_setting_tuple
    data_path, file_to_encode, save_name = encoding_data_tuple

    path_A_to_B = os.path.join(data_path, file_to_encode + 'A_to_B.csv')
    path_B_to_A = os.path.join(data_path, file_to_encode + 'B_to_A.csv')

    save_path_A_to_B = os.path.join(data_path, save_name + 'A_to_B.csv')
    save_path_B_to_A = os.path.join(data_path, save_name + 'B_to_A.csv')

    opts = Options(jtvae_path=jtvae_path,
                   hidden_size=hidden_size,
                   latent_size=latent_size,
                   depth=depth,
                   jtnn_model_path=jtnn_model_path,
                   vocab_path=vocab_path)
    model = load_model(opts)

    smiles_A_to_B = decode_from_jtvae(path_A_to_B, opts, model)
    smiles_B_to_A = decode_from_jtvae(path_B_to_A, opts, model)

    df_to_save_A_to_B = pd.DataFrame(smiles_A_to_B, columns=['SMILES'])
    df_to_save_B_to_A = pd.DataFrame(smiles_B_to_A, columns=['SMILES'])

    df_to_save_A_to_B.to_csv(save_path_A_to_B, index=False)
    df_to_save_B_to_A.to_csv(save_path_B_to_A, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jtvae_path", default="./jtvae/")
    parser.add_argument("--jtnn_model_path", default="molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4")
    parser.add_argument("--vocab_path", default="data/zinc/vocab.txt")

    parser.add_argument("--hidden_size", default=450, type=int)
    parser.add_argument("--latent_size", default=56, type=int)
    parser.add_argument("--depth", default=3, type=int)

    parser.add_argument("--data_path", default="./data/results/aromatic_rings/")
    parser.add_argument("--file_to_encode", default="X_cycle_GAN_encoded_")
    parser.add_argument("--save_name", default="smiles_list_")

    args = parser.parse_args()

    jtvae_path_tuple = (args.jtvae_path, args.jtnn_model_path, args.vocab_path)
    jtvae_setting_tuple = (args.hidden_size, args.latent_size, args.depth)
    encoding_data_tuple = (args.data_path, args.file_to_encode, args.save_name)

    decode(jtvae_path_tuple,
           jtvae_setting_tuple,
           encoding_data_tuple)


if __name__ == "__main__":
    main()
