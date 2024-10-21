from __future__ import annotations

import logging
from multiprocessing.spawn import _main
import pprint
import re
from typing import Dict, Tuple

from rdkit import Chem
# import datamol as dm
from CombineMols.CombineMols import CombineMols
from rdkit.Chem import Draw

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions

import time
import os
from PIL import Image
import pandas as pd
import csv
import os.path as osp




DrawingOptions.bondLineWidth=1.8
DrawingOptions.atomLabelFontSize=14
DrawingOptions.includeAtomNumbers=True

class SmilesRepeat():
    def __init__(self, repeat_times, task_name='tg',root='data_pyg/prediction/') -> None:
        self.repeat_times = repeat_times
        self.task_name = task_name
        self.root = osp.join(root,task_name)
        self.raw_file = osp.join(root,task_name,'1','raw.csv')

    def get_connection_info(self, mol=None, symbol="*") -> Dict:
        """Get connection information a PSMILES string.

        Args:
            mol (Chem.RWMol, optional): _description_. Defaults to None.
            symbol (str, optional): _description_. Defaults to "*".

        Returns:
            Dict: Dictionary containing information of the mol
        """   

        ret_dict = {}

        stars_indices, stars_type, all_symbols, all_index = [], [], [], []
        for star_idx, atom in enumerate(mol.GetAtoms()):
            all_symbols.append(atom.GetSymbol())
            all_index.append(atom.GetIdx())
            if symbol in atom.GetSymbol():
                stars_indices.append(star_idx)
                stars_type.append(atom.GetSmarts())

        num_of_stars = len(stars_indices)
        if num_of_stars < 2:
            return {}
        
        stars_bond = mol.GetBondBetweenAtoms(stars_indices[0], stars_indices[1])
        if stars_bond:
            stars_bond = stars_bond.GetBondType()

        ret_dict["symbols"] = all_symbols
        ret_dict["index"] = all_index

        ret_dict["star"] = {
            "index": stars_indices,
            "atom_type": stars_type,
            "bond_type": stars_bond,
        }

        # multiple neighbors are possible
        neighbor_indices = []
        for i in range(num_of_stars):
            neighbor_indices.append([x.GetIdx() for x in mol.GetAtomWithIdx(stars_indices[i]).GetNeighbors()])
            #[x.GetIdx() for x in mol.GetAtomWithIdx(stars_indices[1]).GetNeighbors()],

        neighbors_type = []
        for i in range(num_of_stars):
            neighbors_type.append([mol.GetAtomWithIdx(x).GetSmarts() for x in neighbor_indices[0]])

        # Bonds between stars and neighbors
        neighbor_bonds = []
        for i in range(num_of_stars):
            neighbor_bonds.append([mol.GetBondBetweenAtoms(stars_indices[i], x).GetBondType()
                                for x in neighbor_indices[i]])
                                
        s_path = None
        if neighbor_indices[0][0] != neighbor_indices[1][0]:
            s_path = Chem.GetShortestPath(
                mol, neighbor_indices[0][0], neighbor_indices[1][0]
            )

        ret_dict["neighbor"] = {
            "index": neighbor_indices,
            "atom_type": neighbors_type,
            "bond_type": neighbor_bonds,
            "path": s_path,
        }

        # Stereo info
        stereo_info = []
        for b in mol.GetBonds():
            bond_type = b.GetStereo()
            if bond_type != Chem.rdchem.BondStereo.STEREONONE:
                idx = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
                neigh_idx = b.GetStereoAtoms()
                stereo_info.append(
                    {
                        "bond_type": bond_type,
                        "atom_idx": idx,
                        "bond_idx": b.GetIdx(),
                        "neighbor_idx": list(neigh_idx),
                    }
                )

        ret_dict["stereo"] = stereo_info

        # Ring info
        ring_info = mol.GetRingInfo()
        ret_dict["atom_rings"] = ring_info.AtomRings()
        ret_dict["bond_rings"] = ring_info.BondRings()
        #print(ret_dict)
        return ret_dict

    def get_mol(self, psmiles) -> Chem.RWMol:
        """Returns a RDKit mol object.

        Note:
            In jupyter notebooks, this function draws the SMILES string

        Returns:
            Chem.MolFromSmiles: Mol object
        """
        return Chem.RWMol(Chem.MolFromSmiles(psmiles))

    def edit_mol(self, ori_psmiles, des_psmiles) -> str:
        start_mol, des_mol = self.get_mol(ori_psmiles), self.get_mol(des_psmiles)
        #Draw.MolToFile(start_mol, 'images/start_mol.png')

        # Stitch these together is to make an editable copy of the molecule object
        combo = Chem.CombineMols(start_mol,des_mol)
        comboSmile = Chem.MolToSmiles(combo)
        #print(f"Combo SMILES: {comboSmile}\n")
        #print(Chem.MolToMolBlock(combo))
        
        # Obtain connection info for future bonds/atoms remove/add
        info = self.get_connection_info(combo)
        if not info:
            print("************************** No Star Mark! **************************")
            return des_psmiles

        # add a new bond between two star symbols and discard these two stars
        edcombo = Chem.EditableMol(combo)
        staridx1, staridx2 = 0, -1
        edcombo.AddBond(
                info["neighbor"]["index"][staridx1][0],
                info["neighbor"]["index"][staridx2][0],
                info["neighbor"]["bond_type"][staridx1][0],
            )
        edcombo.RemoveBond(info["star"]["index"][staridx1], info["neighbor"]["index"][staridx1][0])
        edcombo.RemoveBond(info["star"]["index"][staridx2], info["neighbor"]["index"][staridx2][0])
        edcombo.RemoveAtom(info["star"]["index"][staridx2])
        edcombo.RemoveAtom(info["star"]["index"][staridx1])
        back = edcombo.GetMol()
        backSmile = Chem.MolToSmiles(back)
            #print(f"Back SMILES: {backSmile}\n")
        return backSmile
        
    def star_edge(self, ori_psmiles) -> str:
        ori_mol = self.get_mol(ori_psmiles)
        info = self.get_connection_info(ori_mol)
        if not info or not info["neighbor"]['path']:
            print(f"************************** No Star Mark: {ori_psmiles} **************************")
            return ori_psmiles

        edsmiles = Chem.EditableMol(ori_mol)
        staridx1, staridx2 = 0, -1
        # see if the neighbors of stars are already bonded with each other
        # can not replace star as an edge
        neighidx1, neighidx2 = info["neighbor"]["index"][staridx1][0],info["neighbor"]["index"][staridx2][0]
        path = list(info["neighbor"]['path'])
        for i in range(len(path)):
            if i < len(path) - 1:
                if (path[i] == neighidx1 and path[i+1] == neighidx2) or (path[i] == neighidx2 and path[i+1] == neighidx1):
                    # Instead of removing atoms, we'll replace the star atoms with carbon atoms
                    rwmol = Chem.RWMol(ori_mol)
                    rwmol.GetAtomWithIdx(info["star"]["index"][staridx1]).SetAtomicNum(6)  # 6 is the atomic number for carbon
                    rwmol.GetAtomWithIdx(info["star"]["index"][staridx2]).SetAtomicNum(6)
                    back = rwmol.GetMol()
                    backSmile = Chem.MolToSmiles(back)
                    print(f"************************** Warning: Stars replaced with carbon atoms **************************")
                    print(f"Original SMILES: {ori_psmiles}")
                    print(f"Resulting SMILES: {backSmile}")
                    return backSmile
        else:
            edsmiles.AddBond(
                    info["neighbor"]["index"][staridx1][0],
                    info["neighbor"]["index"][staridx2][0],
                    info["neighbor"]["bond_type"][staridx1][0],
                )
            edsmiles.RemoveBond(info["star"]["index"][staridx1], info["neighbor"]["index"][staridx1][0])
            edsmiles.RemoveBond(info["star"]["index"][staridx2], info["neighbor"]["index"][staridx2][0])
            edsmiles.RemoveAtom(info["star"]["index"][staridx2])
            edsmiles.RemoveAtom(info["star"]["index"][staridx1])
            back = edsmiles.GetMol()
            backSmile = Chem.MolToSmiles(back)
                #print(f"Back SMILES: {backSmile}\n")
            return backSmile


    def direct_edit_mol(ori_psmiles, des_psmiles) -> str:
        ori_psmiles = ori_psmiles.replace("*", "I")
        des_psmiles = ori_psmiles
        mol_rep = CombineMols(ori_psmiles, des_psmiles, "I")
        backSmile = []
        seen = set()
        for i in range(len(mol_rep)):
            j = Chem.MolToSmiles(mol_rep[i])
            if j not in seen:
                seen.add(j)
                moll = j.replace("I","*")
                print(f"SMILES: {moll}\n")
                backSmile.append(moll)
                Draw.MolToFile(mol_rep[i], 'direct_edit_%s.png'%moll)
        
        return backSmile

    def dfs(self, psmiles, n) -> str:
        n = int(n)
        if n == 2:
            mol = self.edit_mol(psmiles, psmiles)
            return mol
        elif n == 1:
            return psmiles
        else:
            if n%2 != 0:
                return self.edit_mol(self.dfs(psmiles, n-1), psmiles)
            else:
                return self.edit_mol(self.dfs(psmiles, n//2), self.dfs(psmiles, n//2))

    def repeat(self):
        '''
        mol = get_mol(res)
        # Use direct edit
        # Draw.MolToFile(mol, newpath+'/d_%s.png'%i)
        Draw.MolToFile(mol, 'res_img/polymers/%s_res.png'%ocur_times)
        '''
        if not osp.exists(self.raw_file):
            print("************************** Raw File Doesn't Exist! **************************")
        else:
            df = pd.read_csv(self.raw_file)     
            new_root = '{}'.format(self.repeat_times)
            path1 = osp.join(self.root, new_root)
            if not osp.exists(path1):
                os.makedirs(path1)
            newcsvpath = osp.join(path1,'raw.csv')
            if not osp.exists(newcsvpath):
                with open(newcsvpath, 'w') as csv_out_file:
                    writer = csv.writer(csv_out_file)
                    # Write Columns_name
                    writer.writerow(df.columns.values.tolist())
                    if self.repeat_times == 0:
                        for r in range(df.shape[0]):
                            init = df['SMILES'][r]
                            res = self.star_edge(init)
                            if res:
                                writer.writerow([res,df[self.task_name][r]])
                    else:
                        for r in range(df.shape[0]):
                            init = df['SMILES'][r]
                            res = self.dfs(init, self.repeat_times)
                            if res:
                                writer.writerow([res,df[self.task_name][r]])

                print("************************** Editing is done! **************************")
            else:
                print("************************** File Existed! **************************")


if __name__=='__main__':
    pass
