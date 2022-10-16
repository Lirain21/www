from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.datasets import fetch_species_distributions

import torch
import numpy as np

def calculate_fingerprint(data):
    smi_list = data.smiles
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)
    
    # change the different fingerprint
    # fps = [Chem.RDKFingerprint(x) for x in mol_list]
    # get the MACCS 
    fps = [MACCSkeys.GenMACCSKeys(x) for x in mol_list]
    # get the ECFPs of the first kind
    # fps = [AllChem.GetMorganFingerprint(x,6) for x in mol_list]
    # get the ECFPS of the second kind 
    # fps = [AllChem.GetMorganFingerprintAsBitVect(x, 6, nBits=1024) for x in mol_list]
    # get the FCFPS
    # fps = [AllChem.GetMorganFingerprint(x,6,useFeatures=True) for x in mol_list]

    return fps

def calculate_similarity(s_data, q_data):
    fps_s = calculate_fingerprint(s_data) # list
    fps_q = calculate_fingerprint(q_data)
    q_len = int(len(q_data.y))
    s_len = int(len(s_data.y))
    # fps_s = torch.cat(fps_s, dim=0).unsqueeze(0).repeat(q_len,1,1)
    # fps_q = torch.cat(fps_q, dim=0).unsqueeze(1).repeat(1,s_len,1)
    # fps = torch.cat((fps_s, fps_q), dim=1)
    simi_list = [[[0.00 for i in range(s_len+1)] for j in range(s_len+1)] for z in range(q_len)]
    simi_list = np.array(simi_list)
    for i in range(q_len):
        fps_s.append(fps_q[i])
        for z in range(s_len+1):
            for y in range(s_len+1):
                # sm01=DataStructs.FingerprintSimilarity(fps_s[z],fps_s[y])
                sm01 = DataStructs.DiceSimilarity(fps_s[z], fps_s[y])
                if z==y:
                    sm01=0
                simi_list[i,z,y] = sm01             
        del fps_s[-1]  
    return simi_list
                

    


