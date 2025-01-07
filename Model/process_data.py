import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
from utils import get_MACCS
from drug_util import drug_feature_extract



def getData(dataset):
    if dataset == 'ALMANAC_Loewe':
        drug_smiles_file = '../../Data/ALMANAC_Loewe/drug_smiles.csv'
        cline_feature_file = '../../Data/ALMANAC_Loewe/cell line_gene_expression.csv'
        drug_synergy_file = '../../Data/ALMANAC_Loewe/drug_synergy.csv'
    elif dataset == 'DrugCombDB_ZIP':
        drug_smiles_file = '../../Data/DrugCombDB/drug_smiles.csv'
        cline_feature_file = '../../Data/DrugCombDB/cell line_gene_expression.csv'
        drug_synergy_file = '../../Data/DrugCombDB/drug_synergy_ZIP.csv'
    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drug_data = []
    drug_smiles_fea = []
    featurizer = dc.feat.ConvMolFeaturizer()
    for tup in zip(drug['pubchemid'], drug['isosmiles']):
        mol = Chem.MolFromSmiles(tup[1])
        mol_f = featurizer.featurize(mol)
        drug_data.append([mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()])
        drug_smiles_fea.append(get_MACCS(tup[1]))
    drug_fea = drug_feature_extract(drug_data)
    num_drug = len(drug_fea)
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    cline_fea = np.array(gene_data, dtype='float32')
    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    synergy = [[int(row[1]), int(row[2]), int(row[3]+num_drug), float(row[4])] for index, row in
               synergy_load.iterrows()]
    return cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy

if __name__ == '__main__':
    cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy = getData("ALMANAC_Loewe")
    # cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy = getData('DrugCombDB_ZIP')
    print(synergy[:10])

