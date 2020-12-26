import os
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch

import matplotlib.pyplot as plt
from collections import Counter # Del

def read_data(path='data/hepatotoxicity.txt', min_atom=None, max_atom=None):
    '''Load toxicity dataset'''
    print(f'Loading {path}...')

    with open(path, 'r') as f:
        lines = f.readlines()

    tox_szs = [] # Del
    ntox_szs = []
    mols, toxs, reliabilities = [], [], []
    for i, line in enumerate(lines):
        _, s, t, r = line.split()
        
        mol = Chem.MolFromSmiles(s, True)
        if min_atom is None and max_atom is None:
            raise ValueError('Provide either min_atom or max_atom')
        if mol is None or mol.GetNumAtoms() >= max_atom or mol.GetNumAtoms() <= min_atom: 
            continue 
        else:
            if t == '1': # Del
                tox_szs.append(mol.GetNumAtoms())
            else:
                ntox_szs.append(mol.GetNumAtoms())
            mols.append(Chem.MolFromSmiles(s, True))
            toxs.append(t == '1')

    hist = plt.hist(ntox_szs, bins=45, density=True)
    plt.show()
    
    data = []
    for m, t in zip(mols, toxs):
        data.append({'x': get_atom_feature(m),
                     'adj': get_adj_matrix(m),
                     'y': t})

    print(f"- data length: {len(data)} / {len(lines)}")
    print(f"- percent toxic: {sum(toxs)/len(toxs):.3f}")
    print(f"- atomic feature size: {data[0]['x'].shape[1]}")

    return data


def onehot_encode(x, xset):
    if x not in xset:
        print(f'Missing: {x}')
        x = xset[-1]
    return list(map(lambda s: x == s, xset))


def get_atom_feature(mol):
    '''Get atom features: Symbol, Degree'''
    atom_features = []
    for atom in mol.GetAtoms():
        f = onehot_encode(atom.GetSymbol(),
                         ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                         'Na', 'Ca', 'Cr', 'Hg', 'K', 'Sn', 'Si', 'Al', 'Mn', 'V',
                         'Ge', 'As', 'Ag', 'Zn', 'B', 'Ni', 'Ti', 'Fe', 'Mg', 'X']) + \
            onehot_encode(atom.GetDegree(),
                         [0, 1, 2, 3, 4, 5]) + \
            onehot_encode(atom.GetFormalCharge(), [-1, 0, 1, 2, 3, 4]) + \
            onehot_encode(atom.GetImplicitValence(), [0, 1, 2, 3]) + \
            onehot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + \
            onehot_encode(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                                                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                                                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                                                Chem.rdchem.ChiralType.CHI_OTHER]) + \
            onehot_encode(atom.GetHybridization(), [Chem.rdchem.HybridizationType.S,
                                                    Chem.rdchem.HybridizationType.SP,
                                                    Chem.rdchem.HybridizationType.SP2,
                                                    Chem.rdchem.HybridizationType.SP3,
                                                    Chem.rdchem.HybridizationType.SP3D,
                                                    Chem.rdchem.HybridizationType.SP3D2]) + \
            [atom.IsInRingSize(3)] + [atom.IsInRingSize(4)] + \
            [atom.IsInRingSize(5)] + [atom.IsInRingSize(6)] + \
            [atom.IsInRingSize(7)] + [atom.IsInRingSize(8)] + \
            [atom.GetIsAromatic()]      

        atom_features.append(f)
    
    atom_features = np.array(atom_features)
    return atom_features


# def get_atom_feature(mol):
#     '''Get atom features: Symbol, Degree'''
#     atom_features = []
#     for atom in mol.GetAtoms():
#         f = onehot_encode(atom.GetSymbol(),
#                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'X']) + \
#             onehot_encode(atom.GetDegree(),
#                          [0, 1, 2, 3, 4, 5, 'X'])
#             
#         atom_features.append(f)
#     
#     atom_features = np.array(atom_features)
#     return atom_features
 

def get_adj_matrix(mol):
    ''''Get self-loop added adjacency matrix'''
    n = mol.GetNumAtoms()
    adj_matrix = GetAdjacencyMatrix(mol) + np.eye(n)
    
    adj_matrix = np.array(adj_matrix)
    return adj_matrix


def check_dimension(objs):
    '''Get max size'''
    sizes = []
    for obj in objs:
        if isinstance(obj, np.ndarray):
            sizes.append(obj.shape)
        else:
            sizes.append(0)

    sizes = np.array(sizes)
    max_size = np.max(sizes, 0)
    return max_size


def collate_obj(obj, max_obj, idx):
    '''Collate function for most data type'''
    if isinstance(obj, np.ndarray):
        dims = obj.shape
        max_dims = max_obj.shape

        slices = tuple([slice(0, dim) for dim in dims])
        slices = [slice(idx, idx+1), *slices]
        max_obj[tuple(slices)] = obj
    else:
        max_obj[idx] = obj
    
    return max_obj


def dict_collate_fn(batch_dict):
    '''Collate function for batch of dicts'''
    all_items = [i for e in batch_dict for i in e.items()]
    all_keys, all_values = list(zip(*all_items))

    batch_size = len(batch_dict)
    num_element = int(len(all_items)/batch_size)
    all_keys = all_keys[0:num_element]

    dim_dict = {}
    for i, key in enumerate(all_keys):
        values = [v for j, v in enumerate(all_values) if j % num_element == i]
        if isinstance(values[0], np.ndarray):
            dim_dict[key] = np.zeros(np.array([batch_size, *check_dimension(values)]))
        elif isinstance(values[0], str):
            dim_dict[key] = ["" for _ in range(batch_size)]
        else:
            dim_dict[key] = np.zeros([batch_size,])
    
    collate_dict = {}
    for i in range(batch_size):
        if batch_dict[i] == None: continue
        keys =[]
        for key, value in dim_dict.items():
            value = collate_obj(batch_dict[i][key], value, i)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            collate_dict[key] = value
    
    return collate_dict


def dict_to_device(batch_dict, device):
    '''Value in dict to device, cpu or cuda'''
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            batch_dict[key] = value
    
    return batch_dict
        

if __name__ == '__main__':
    data = read_data(min_atom=5, max_atom=50)
