#!/usr/bin/python
# -*- coding:utf-8 -*-
from rdkit import Chem
from typing import List, Tuple
from data.format import Block, Atom, VOCAB
from p_tqdm import p_map
from .rdkit_to_blocks import rdkit_to_blocks

def sdf_to_list_blocks(sdf_file: str, using_hydrogen: bool = False, dict_form: bool=False, silent: bool=False) -> List[Tuple[List[Block], str]]:
    '''
        Convert an SDF file to a list of lists of blocks for each molecule in parallel.
        
        Parameters:
            sdf_file: Path to the SDF file
            using_hydrogen: Whether to preserve hydrogen atoms, default false
            dict_form: Return a dict where the keys are the names of the molecules

        Returns:
            A list of lists of blocks and SMILES. Each inner list represents the blocks for a molecule.
    '''
    # Read SDF file
    supplier = Chem.SDMolSupplier(sdf_file)

    # Define function to process a single molecule
    def process_molecule(mol):
        if mol is not None:
            blocks = rdkit_to_blocks(mol, using_hydrogen=using_hydrogen)
            return blocks
        else:
            return None

    # Parallel processing of molecules
    results = p_map(process_molecule, supplier, disable=silent)
    smiles = [Chem.MolToSmiles(mol) if mol is not None else None for mol in supplier]

    if dict_form:
        # Get names
        names = [mol.GetProp('_Name') for mol in supplier]
        results_dict = {}
        for n, r, smi in zip(names, results, smiles):
            if r is None: continue
            results_dict[n] = (r, smi)
        return results_dict
    else:
        # Remove None results
        final_results = []
        for r, smi in zip(results, smiles):
            if r is None: continue
            final_results.append((r, smi))
        # Return the final list of lists of blocks
        return final_results

if __name__ == '__main__':
    import sys
    list_blocks = sdf_to_list_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of molecules: {len(list_blocks)}')
