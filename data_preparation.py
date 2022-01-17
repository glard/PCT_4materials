#!/usr/bin/env python
# coding: utf-8


# Developed 07/13/2020 by Jeffrey Hu with assistance from Professor Jianjun Hu
# Sorts materials cif files by element number and space group
from pymatgen.io.cif import CifParser

from pymatgen.core.composition import Composition


import numpy as np

from pymatgen.core.structure import Structure

import multiprocessing as mp
import pandas as pd
import h5py

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
print(len(elements))

# read cif structure of .cif file
structure_from_cif = Structure.from_file('data/CIF-DATA/mp-27936.cif')

# get coordinates
cartesian_cords = structure_from_cif.cart_coords
print(f"cartesian cords is {cartesian_cords}")
print(f"cartesian cords shape is {cartesian_cords.shape}")

df = pd.read_csv('data/properties-reference/bandgap.csv', header= None)
files = df.values[:, 0]

print(len(files))

#ply_data_train0 = pd.read_hdf('data/modelnet40_ply_hdf5_2048/ply_data_train0.h5')
# hf = h5py.File('data/modelnet40_ply_hdf5_2048/ply_data_train0.h5', 'r')

filename = 'data/modelnet40_ply_hdf5_2048/ply_data_test0.h5'
with h5py.File(filename, "r") as f:
    # ar = f.get('dataset_name').value
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]
    c_group_key = list(f.keys())[2]
    d_group_key = list(f.keys())[3]

    # Get the data
    data = (f[a_group_key])
    print(data.shape)
    print(data.dtype)
    print(data[0].shape)
    # faceId
    # faceId = (f[b_group_key])
    # print(faceId.shape)
    # print(faceId.dtype)
    # print(faceId[0])
    # label
    label = (f[c_group_key])
    print(label.shape)
    print(label.dtype)
    print(label[0])
    print(label[1])
    # normal
    # normal = (f[d_group_key])
    # print(normal.shape)
    # print(normal.dtype)


print()
def base(file):
    crystal = Structure.from_file('data/CIF-DATA/%s.cif' % file)
    shortestDistance = {}
    for i, x1 in enumerate(crystal.sites):
        for j, y1 in enumerate(crystal.sites):
            if i == j or i > j:
                continue
            # print(x1.specie)
            x = str(x1.specie)
            y = str(y1.specie)
            if not (x, y) in shortestDistance:
                shortestDistance[(x, y)] = crystal.distance_matrix[i, j]
            else:
                if crystal.distance_matrix[i, j] < shortestDistance[(x, y)]:
                    shortestDistance[(x, y)] = crystal.distance_matrix[i, j]
    return shortestDistance


pool = mp.Pool(processes=6)
results = [pool.apply_async(base, args=(f,)) for f in files]


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
# This returns a tensor
inputs = Input(shape=(4,))
# A layer instance is callable on a tensor, and returns a tensor
x = Dense(5, activation='relu')(inputs)
x = Dense(10, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)
# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=outputs)
model.summary()