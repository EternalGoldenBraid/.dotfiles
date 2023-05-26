# Import the necessary libraries
from rdkit import Chem
from rdkit.Chem import Draw

# Read in the molecule from a file or create it using a molecule builder
mol = Chem.MolFromSmiles('C10H12O')

# Generate a graph representation of the molecule
graph = Chem.rdmolops.GetMolGraph(mol)

# Use the Draw.MolToImage function to generate an image of the molecule
img = Draw.MolToImage(mol, size=(400,400))

# Save the image to a file or display it using matplotlib
img.save('cinnamon.png')
