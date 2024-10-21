from rdkit import Chem
import datamol as dm
from CombineMols.CombineMols import CombineMols
from rdkit.Chem import Draw    


def strtoimg(smiles):
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    # Use direct edit
    #Draw.MolToFile(mol, newpath+'/d_%s.png'%i)
    Draw.MolToFile(mol, '{}.png'.format(smiles[:6]))

if __name__ == '__main__':
    strtoimg('*CCC(=O)Nc1ccc(NC(=O)CCN2C(=O)c3ccc(C(=O)c4ccc5c(c4)C(=O)N(*)C5=O)cc3C2=O)cc1')