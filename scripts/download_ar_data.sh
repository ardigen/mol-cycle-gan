#!/bin/sh

cd data/input_data/aromatic_rings

wget http://molcyclegan.ardigen.com/X_JTVAE_zinc_train_A.csv
wget http://molcyclegan.ardigen.com/X_JTVAE_zinc_train_B.csv
wget http://molcyclegan.ardigen.com/X_JTVAE_zinc_test_A.csv
wget http://molcyclegan.ardigen.com/X_JTVAE_zinc_test_B.csv


cd ../../
mkdir results
cd results
mkdir aromatic_rings
cd aromatic_rings

wget http://molcyclegan.ardigen.com/smiles_list_A_to_B.csv
wget http://molcyclegan.ardigen.com/smiles_list_B_to_A.csv
wget http://molcyclegan.ardigen.com/X_cycle_GAN_encoded_A_to_B.csv
wget http://molcyclegan.ardigen.com/X_cycle_GAN_encoded_B_to_A.csv
