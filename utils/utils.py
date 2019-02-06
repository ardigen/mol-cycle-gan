import os
from functools import wraps

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Lipinski import NumAromaticRings


def try_or_none(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return None

    return wrapper


def smiles_validity_check(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def canon_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    else:
        return None


@try_or_none
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


@try_or_none
def get_number_of_aromatic_rings(smi):
    mol = Chem.MolFromSmiles(smi)
    return NumAromaticRings(mol)


def get_smiles_number_of_aromatic_rings(smiles_list, delete_nans=False):
    smiles_AR = [get_number_of_aromatic_rings(smi) for smi in smiles_list]

    if delete_nans:
        smiles_AR = [x for x in smiles_AR if x is not None]

    return smiles_AR


def create_tanimoto_column(smiles_A, smiles_B):
    df_smiles = pd.DataFrame({'A': smiles_A, 'B': smiles_B})
    df_smiles = df_smiles.iloc[np.logical_and(df_smiles['A'].values != 'nan',
                                              df_smiles['B'].values != 'nan')]

    df_smiles.dropna(inplace=True)
    smiles_A = df_smiles.A
    smiles_B = df_smiles.B

    smiles_A_mol = [Chem.MolFromSmiles(x) for x in smiles_A]
    smiles_B_mol = [Chem.MolFromSmiles(x) for x in smiles_B]

    smiles_A_fps = [AllChem.GetMorganFingerprint(mol, 2) for mol in smiles_A_mol]
    smiles_B_fps = [AllChem.GetMorganFingerprint(mol, 2) for mol in smiles_B_mol]

    tanimoto = np.array([DataStructs.TanimotoSimilarity(fp1, fp2)
                         for (fp1, fp2)
                         in zip(smiles_A_fps, smiles_B_fps)])

    return tanimoto


def crate_df_with_best_scores(smiles_A, smiles_B):
    df_smiles = pd.DataFrame({'A': smiles_A, 'B': smiles_B})
    df_smiles.dropna(inplace=True)
    df_smiles = df_smiles.iloc[np.logical_and(df_smiles['A'].values != 'nan',
                                              df_smiles['B'].values != 'nan')]

    df_smiles.dropna(inplace=True)
    smiles_A = df_smiles.A
    smiles_B = df_smiles.B

    score_A = np.array(get_smiles_number_of_aromatic_rings(smiles_A, delete_nans=False))
    score_B = np.array(get_smiles_number_of_aromatic_rings(smiles_B, delete_nans=False))

    df_constrained = pd.DataFrame({'A': smiles_A,
                                   'B': smiles_B,
                                   'scores_A': score_A,
                                   'scores_B': score_B,
                                   'same_aromatic_number': score_B == score_A})
    return df_constrained


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
