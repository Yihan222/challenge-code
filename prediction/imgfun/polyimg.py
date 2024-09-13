from rdkit import Chem
import datamol as dm
from CombineMols.CombineMols import CombineMols
from rdkit.Chem import Draw    


def strtoimg(smiles):
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    # Use direct edit
    #Draw.MolToFile(mol, newpath+'/d_%s.png'%i)
    Draw.MolToFile(mol, './res_img/polymers/%s_res.png'%smiles)

if __name__ == '__main__':
    strtoimg('[H]C1CC(*)CC1\C=C\*')