import torch
import os
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.nn import radius_graph
from model.model import O3Transformer
from model.norms import EquivariantLayerNorm
from e3nn import o3
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import numpy as np
import pickle
import MDAnalysis as mda

import warnings
warnings.filterwarnings("ignore")

types = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br':9}
bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
hybid = {HybridizationType.UNSPECIFIED: 0,HybridizationType.S:1, HybridizationType.SP: 2, HybridizationType.SP2: 3 ,   HybridizationType.SP3: 4,
HybridizationType.SP3D:5 }


class Predictor:
    def __init__(self, model: torch.nn.Module, pred_data: DataLoader) -> None:
        super().__init__()
        self.gpu_id = 0
        self.model = model.to(self.gpu_id)
        self.pred_data = pred_data


    def predict(self, pred_data: Data, message: str, mean: float, std: float):
        print(message, '...')
        torch.cuda.empty_cache()
        criterion = torch.nn.L1Loss()
    
        nmr_trues = []
        nmr_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, loader_dict in enumerate(pred_data):
                    N = loader_dict["n_mols"]
                    N_nmr = loader_dict["n_nmr"]
                    for data in loader_dict["loader"]:
                            data = data.to(self.gpu_id) 
                            c_mask = data.x[:,0] == 2.0
                            nmr_mask = data.x[:,-1] > -0.5
                            mask = nmr_mask.logical_and(c_mask) 
                            nmr_true =  data.x[:,-1]
                            nmr_masked = nmr_true[mask]
                            out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                                        edge_index = data.edge_index, edge_attr = data.edge_attr.long(), batch = data.batch)
                            out_masked = out[mask]*std + mean
                            
                            out = (out_masked.reshape(N,N_nmr).T).mean(dim = 1)
                            trues = nmr_masked[:N_nmr].detach().flatten()
                            nmr_trues.append(trues)
                            nmr_preds.append(out.detach().flatten())
        return nmr_preds

def load_pred_data(pred_data_path: str):
    with open(pred_data_path, 'rb') as handle:
        pred_data = pickle.load(handle)
    pred_data_ = InMemoryDataset()
    pred_data_.data, pred_data_.slices = pred_data_.collate(pred_data)
    return pred_data_

def prepare_dataloader(dataset: InMemoryDataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

def get_cut_off_graph(edge_index, edge_attr, p, cut_off=6.0):
    row, col = edge_index
    dist = torch.sqrt(torch.sum((p[row] - p[col])**2, dim=1))
    mask = dist <= cut_off
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce  = "max")
    return edge_index, edge_attr

def generate_conformations(m_, nbr_confs):
    params = Chem.rdDistGeom.ETKDGv3()
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = 0.001
    params.numThreads = 10
    params.enforceChirality = True
    params.maxAttempts = 10000
    Chem.SanitizeMol(m_)
    m_ = Chem.AddHs(m_, addCoords=True)

    em = Chem.rdDistGeom.EmbedMultipleConfs(m_, numConfs=nbr_confs*2, params=params)

    ps = AllChem.MMFFGetMoleculeProperties(m_, mmffVariant='MMFF94')

    energies = []
    confs = []
    for conf in m_.GetConformers():
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m_, ps, confId=conf.GetId())

        if isinstance(ff, type(None)):
            continue
        energy = ff.CalcEnergy()
        energies.append(energy)
        confs.append(conf)

    m_ = Chem.RemoveHs(m_)
    if em == -1:
        conformations = []
        for i, c in enumerate(m_.GetConformers()):
            xyz = c.GetPositions()
            conformations.append(xyz)
            if i > 9:
                return conformations
    energies = np.array(energies)
    ind = energies.argsort()[:nbr_confs]
    energies = energies[ind]
    conformations = []
    for i, c in enumerate(m_.GetConformers()):

        if i not in ind:
            continue
        xyz = c.GetPositions()
        conformations.append(xyz)



    return conformations, energies, m_

def process_mol(smiles_list, mol_name):
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
        spectrum_str_13C = "{"
        spectrum_str_1H = "{"
        for idx in carbon_indices:
            spectrum_str_13C += f"{idx}: {00.0}, "
            spectrum_str_1H += f"{idx}: [{0.00},{0}], "
        spectrum_str_13C.rstrip(", ")
        spectrum_str_1H.rstrip(", ")
        spectrum_str_13C += "}"
        spectrum_str_1H += "}"
        print(spectrum_str_13C)
        print(spectrum_str_1H)
        mol.SetProp("13C Spectrum", spectrum_str_13C)
        mol.SetProp("1H Spectrum", spectrum_str_1H)
        mols.append(mol)

    type_idx_list = []
    num_hs = []
    nmr_list = []
    nmr_spec = mol.GetProp("13C Spectrum")
    nmr_spec = eval(nmr_spec)

    for atom in mol.GetAtoms():
        type_idx_list.append(types[atom.GetSymbol()])
        num_hs.append(atom.GetTotalNumHs())
        if atom.GetIdx() in nmr_spec:
            nmr_list.append(nmr_spec[atom.GetIdx()])
        else:
            nmr_list.append(-1)

    type_idx = torch.tensor(type_idx_list, dtype=torch.float).reshape(-1,1)
    num_hs = torch.tensor(num_hs, dtype=torch.float).reshape(-1,1)
    nmr_list = torch.tensor(nmr_list, dtype=torch.float).reshape(-1,1)


    x = torch.cat([type_idx, num_hs, nmr_list], dim=-1)

    row, col = [], []
    bond_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start]
        col += [end]
        bond_attr +=  [bonds[bond.GetBondType()]]
    bond_attr = torch.tensor(bond_attr, dtype=torch.long).flatten()
    edge_index_b = torch.tensor([row, col], dtype=torch.long)
    edge_index_b, bond_attr = to_undirected(edge_index_b,bond_attr)
    
    
    positions = torch.zeros((x.size(0),3))

    positions = torch.tensor(positions, dtype = torch.float)
    edge_index_r = radius_graph(positions, r=1000.)
    edge_index_r = to_undirected(edge_index_r)
    edge_attr_r = torch.zeros(edge_index_r.size(1))
    edge_index = torch.column_stack([edge_index_b,edge_index_r])
    edge_attr = torch.cat([bond_attr, edge_attr_r])
    edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='max')

    dataset = []
    data = Data()
    data.x = x
    data.edge_attr = edge_attr
    data.edge_index = edge_index
    data.pos = positions
    data.name = mol_name
    data.smiles = Chem.MolToSmiles(mol)
    dataset.append(data)

    return mols, dataset, carbon_indices


def create_conformer(mols, mol_name, nbr_confs):
    for mol in mols:
        name = mol_name

        path = "predict//" + name + "_conformations.pickle"
        if os.path.isfile(path):
            continue
        conformations, _,_ = generate_conformations(mol,nbr_confs)
        with open(path, 'wb') as handle:
            pickle.dump(conformations, handle)

def add_conformations_to_predict_dataset(dataset):
    pred_datas = []
    for data in dataset:
        datas_ = []
        name = data.name
        n_atoms = data.x.size(0)
        conf_path = "predict//" + name + "_conformations.pickle"
        k = 0
        if os.path.exists(conf_path):
                with open(conf_path, 'rb') as handle:
                        distencies = pickle.load(handle)
                for j, dis_ in enumerate(distencies):
                        d = data.clone()
                        
                        pos = torch.from_numpy(dis_).float()
                        d.pos = pos
                        edge_index, edge_attr = get_cut_off_graph(d.edge_index, d.edge_attr, d.pos ,cut_off=6.)
                        d.edge_index = edge_index
                        d.edge_attr = edge_attr
                        datas_.append(d)
                        k = k + 1
        else:
                continue
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)
        pred_datas.append( {"name":data.name,"n_atoms": n_atoms, "n_nmr":torch.sum(mask).item(), "n_mols": k,
                                        "loader" : DataLoader(datas_, batch_size = len(datas_), shuffle=False )})
    return pred_datas


def mols_to_pdb(mols, pdb_path):
    for mol in mols:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.MolToPDBFile(mol, pdb_path)

def add_shifts_to_bfactor(pdb_path, output_path, carbon_indices, carbon_shifts, hydrogen_shifts):
    u = mda.Universe(pdb_path)
    atoms = u.atoms
    if '13C' in output_path:
        for idx, carbon_shift in zip(carbon_indices, carbon_shifts):
            atoms[idx].bfactor = carbon_shift.item()
    else:
        for idx, hydrogen_shift in zip(carbon_indices, hydrogen_shifts):
            atoms[idx].bfactor = hydrogen_shift.item()

    with mda.Writer(output_path) as W:
        W.write(u.atoms)

def main(smiles_list: list, mol_name: str, checkpoint_path_13C: str, checkpoint_path_1H: str, batch_size: int, nbr_confs: int):
    np.random.seed(0)
    torch.manual_seed(0)

    mols, dataset, carbon_indices = process_mol(smiles_list, mol_name)

    if not os.path.exists("predict"):  
        os.makedirs("predict") 
    
    
    create_conformer(mols, mol_name, nbr_confs)
    pred_data = add_conformations_to_predict_dataset(dataset)
    pred_data_path = "predict//" + mol_name + "_pred_data.pkl"
    with open(pred_data_path, 'wb') as handle:
        pickle.dump(pred_data, handle)

    pred_data = load_pred_data(pred_data_path)
    pred_loader = prepare_dataloader(pred_data, batch_size)

    model_13C = O3Transformer(norm=EquivariantLayerNorm, n_input=128, n_node_attr=128, n_output=128,
                          irreps_hidden=o3.Irreps("64x0e + 32x1o + 8x2e"), n_layers=7)

    model_1H = O3Transformer(norm=EquivariantLayerNorm, n_input=128, n_node_attr=128, n_output=128,
                          irreps_hidden=o3.Irreps("64x0e + 32x1o + 8x2e"), n_layers=7)

    if os.path.exists(checkpoint_path_13C):
        checkpoint_13C = torch.load(checkpoint_path_13C, weights_only=False)
        model_13C.load_state_dict(checkpoint_13C)
        print(f"Loaded checkpoint from {checkpoint_path_13C}")
    else:
        print(f"Checkpoint {checkpoint_path_13C} not found.")
        return

    if os.path.exists(checkpoint_path_1H):
        checkpoint_1H = torch.load(checkpoint_path_1H, weights_only=False)
        model_1H.load_state_dict(checkpoint_1H)
        print(f"Loaded checkpoint from {checkpoint_path_1H}")
    else:
        print(f"Checkpoint {checkpoint_path_1H} not found.")
        return
    
    predictor_13C = Predictor(model_13C, pred_loader)
    predictor_1H = Predictor(model_1H, pred_loader)
    predictions_13C = predictor_13C.predict(pred_data, 'Predict 13C chemical shifts', mean=73.3624, std=23.0723)
    predictions_1H = predictor_1H.predict(pred_data, 'Predict 1H chemical shifts', mean=3.7660, std=0.7601)

    pred_path_13C = "predict//" + mol_name + "_predictions_13C.pkl"
    with open(pred_path_13C, 'wb') as handle:
        pickle.dump(predictions_13C, handle)
    print("Predictions saved to", pred_path_13C)
    
    pred_path_1H = "predict//" + mol_name + "_predictions_1H.pkl"
    with open(pred_path_1H, 'wb') as handle:
        pickle.dump(predictions_1H, handle)
    print("Predictions saved to", pred_path_1H)

    pdb_path = "predict//" + mol_name + ".pdb"
    mols_to_pdb(mols, pdb_path)
    print("Coordinates saved to", pdb_path)

    for i in ('13C', '1H'):
        pdb_shifts_path = f"predict//{mol_name}_{i}_shifts.pdb"
        add_shifts_to_bfactor(pdb_path, pdb_shifts_path, carbon_indices, predictions_13C[0], predictions_1H[0])
        print("Coordinates with shifts in Bfactor col saved to", pdb_shifts_path)


    print("Predicted chemical shifts:")
    print("13C_idx 13C_CS 1H_CS")
    for i,j,k in zip(carbon_indices, predictions_13C[0], predictions_1H[0]):
        print(f"{i} {j:.2f} {k:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Predict chemical shifts for SMILS molecule using a trained model')
    parser.add_argument('--smiles_list', type=str, nargs='+', help='Molecule SMILES', required=True)
    parser.add_argument('--mol_name', type=str, default='mol1', help='Molecule name')
    parser.add_argument('--nbr_confs', type=int, default=100, help='Number of conformations')
    parser.add_argument('--checkpoint_path_13C', type=str, help='Path to model checkpoint for 13C', required=True)
    parser.add_argument('--checkpoint_path_1H', type=str, help='Path to model checkpoint for 1H', required=True)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for prediction')
    args = parser.parse_args()
    main(args.smiles_list, args.mol_name, args.checkpoint_path_13C, args.checkpoint_path_1H, args.batch_size, args.nbr_confs)

