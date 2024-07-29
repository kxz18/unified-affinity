from openbabel import openbabel as ob
from vina import Vina
import os
import numpy as np
import pickle as pk
from tqdm import tqdm
import multiprocessing as mp

PREPR = '/share/medical_data/downstream/tools/vina/ADFRsuite-1.0/bin/prepare_receptor'

# configs for flexible docking, deprecated
PY27 = None # '/home/jiayinjun/miniconda3/envs/py27/bin/python'
BCL = None # '/home/jiayinjun/flex_dock/bcl-4.3.1-Linux-x86_64/bcl.exe'
ROS = None # '/drug/rosetta.binary.linux.release-315/main'
M2P = None # f'{ROS}/source/scripts/python/public/molfile_to_params.py'
CLP = None # f'{ROS}/tools/protein_tools/scripts/clean_pdb.py'
RSC = None # f'{ROS}/source/bin/rosetta_scripts.static.linuxgccrelease'
#get path of current file
ROP = None # os.path.join(os.path.dirname(os.path.abspath(__file__)),'RosettaLigandOptions.txt')
XML = None # os.path.join(os.path.dirname(os.path.abspath(__file__)),'flexible_docking.xml')

def sdf2pdbqt(sdf_string_list):
    #convert to obabel and get pdbqt
    pdbqt_output_list = []
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("sdf", "pdbqt")
    for sdf_string in sdf_string_list:
        mol = ob.OBMol()
        conv.ReadString(mol, sdf_string)
        pdbqt_output = conv.WriteString(mol)
        pdbqt_output_list.append(pdbqt_output)
    return pdbqt_output_list

def pdbqt2pdb(pdbqt_string_list):
    #convert to obabel and get pdb
    pdb_output_list = []
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("pdbqt", "pdb")
    for pdbqt_string in pdbqt_string_list:
        mol = ob.OBMol()
        conv.ReadString(mol, pdbqt_string)
        pdb_output = conv.WriteString(mol)
        pdb_output_list.append(pdb_output)
    return pdb_output_list

def pdbqt2sdf(pdbqt_string_list):
    #convert to obabel and get sdf
    sdf_output_list = []
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("pdbqt", "sdf")
    for pdb_string in pdbqt_string_list:
        mol = ob.OBMol()
        conv.ReadString(mol, pdb_string)
        sdf_output = conv.WriteString(mol)
        sdf_output_list.append(sdf_output)
    return sdf_output_list

def roslig_ligprep(ligand_name,prot_name):
    os.system(f'''
{BCL} molecule:Filter -add_h -defined_atom_types \
  -3d -input_filenames {prot_name}_{ligand_name}.sdf \
  -output_matched {ligand_name}.CLEANED.sdf \
  -output_unmatched {ligand_name}.UNCLEANED.sdf -message_level Debug >/dev/null 2>&1
wait
{BCL} molecule:ConformerGenerator -rotamer_library cod \
  -top_models 100 -ensemble_filenames {ligand_name}.CLEANED.sdf \
  -conformers_single_file {ligand_name}.CLEANED.conf.sdf \
  -conformation_comparer 'Dihedral(method=Max)' 30 -max_iterations 1000 >/dev/null 2>&1
wait
{PY27} {M2P} -n LIG -p {ligand_name} --chain=X --conformers-in-one-file {ligand_name}.CLEANED.conf.sdf >/dev/null 2>&1
wait
cat {prot_name}_A.pdb {ligand_name}.pdb > {prot_name}_{ligand_name}.pdb 
'''
    )
    #read the last line of {ligand_name}.params
    if os.path.exists(f'{ligand_name}.params'):
        with open(f'{ligand_name}.params','r') as f:
            last_line = f.readlines()[-1]
        assert last_line == f'PDB_ROTAMERS {ligand_name}_conformers.pdb\n', 'ligand params not generated'
    else:
        raise ValueError('ligand params not generated')

def rosetta_docking(ligand_name,prot_name,n_pose):
    os.system(f"{RSC} @{ROP} -s {prot_name}_{ligand_name}.pdb -extra_res_fa {ligand_name}.params -nstruct {n_pose} -parser:protocol {XML} >/dev/null 2>&1")

def vina_scoring(pdbfile,chain='X'):
    #read pdbfiles into lines, seperate chain X and others into two strings
    with open(pdbfile,'r') as f:
        lines = f.readlines()
    chainL = ''
    chainR = ''
    for line in lines:
        if line.startswith('Grid_score'):
            break
        if line[21] == chain:
            chainL += line
        else:
            chainR += line
        
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("pdb", "pdbqt")    
    mol = ob.OBMol()
    conv.ReadString(mol, chainL)
    #get mol coordinates and calculate center
    center = np.array([[atom.GetX(),atom.GetY(),atom.GetZ()] for atom in ob.OBMolAtomIter(mol) if atom.GetAtomicNum() != 1]).mean(axis=0).tolist() #consider only heavy atoms as some carbon have too many hydrogens
    pdbqtL = conv.WriteString(mol)
    pdbR = pdbfile.replace('.pdb','_R.pdb')
    pdbqtR = pdbfile.replace('.pdb','_R.pdbqt')
    with open(pdbR,'w') as f:
        f.write(chainR)
    os.system(f'{PREPR} -r {pdbR} -A "hydrogens" -o {pdbqtR} >/dev/null 2>&1')
    v = Vina(sf_name='vina',cpu=1,verbosity=0)
    v.set_receptor(pdbqtR)
    v.set_ligand_from_string(pdbqtL)
    v.compute_vina_maps(center=center, box_size=[30, 30, 30]) #set box size to 30 angstrom as some ligands are large
    #clean up
    os.remove(pdbR)
    os.remove(pdbqtR)
    # print(f'vina score of {pdbfile}: {v.score()}')
    energy_minimized = v.optimize()
    return energy_minimized[0]

def prep_dock_score(output_dir,name,protein_name,n_flexible):
    clean_up_list = []
    os.chdir(output_dir)
    try:
        ligand_name = name.replace('.sdf','')
        roslig_ligprep(ligand_name,protein_name)
        clean_up_list += [
            os.path.join(output_dir,f'{ligand_name}.CLEANED.sdf'),
            os.path.join(output_dir,f'{ligand_name}.UNCLEANED.sdf'),
            os.path.join(output_dir,f'{ligand_name}.CLEANED.conf.sdf'),
            os.path.join(output_dir,f'{ligand_name}.pdb'),
            os.path.join(output_dir,f'{ligand_name}.params'),
            os.path.join(output_dir,f'{ligand_name}_conformers.pdb'),
            os.path.join(output_dir,f'{protein_name}_{ligand_name}.pdb'),
            os.path.join(output_dir,'score.sc')
        ]
        rosetta_docking(ligand_name,protein_name,n_flexible)
        best_score = None
        best_model = None
        for i in range(n_flexible): 
            complex_pdb = os.path.join(output_dir,f'{protein_name}_{ligand_name}_{i+1:04}.pdb')      
            clean_up_list.append(complex_pdb)
            try:
                score = vina_scoring(complex_pdb)
            except:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_model = complex_pdb
    except:
        print(f'Error in flexible docking ligand {name}')

    for f in clean_up_list:
        if os.path.exists(f) and f != best_model:
            os.remove(f)
    return best_score

def docking_pipeline(
        receptor_pdbqt, #receptor pdbqt file
        ligand_sdf_list, #a list of ligand sdf string with only one molecule in each string
        center, #center of docking box, [x, y, z]
        output_dir, #output directory
        lig_name_list=None, #ligand name list, if None, use LIG1 LIG2 ...
        n_cpu=1, #number of cpu to use
        exhaustiveness=None, #exhaustiveness of docking, higher value will cost more time, default 8 for routine docking, 1 for pose initial of flexible docking, 32 is recommended for high accuracy
        n_rigid=None, #number of rigid docking pose to keep, default 5 for routine docking, 1 for pose initial of flexible docking
        flexible=False, #whether sample receptor flexibility after rigid docking
        receptor_pdb=None, #receptor pdb file, optional, but necessary if flexible=True
        n_flexible=5, #number of flexible docking pose to keep, only useful when flexible=True, in total, n_rigid*n_flexible poses will be kept for each ligand
        verbose=True, # whether to print some information
):
    if receptor_pdbqt is None and receptor_pdb is not None:
        receptor_pdbqt = receptor_pdb.replace('.pdb','.pdbqt')
        # os.system(f'{PREPR} -r {receptor_pdb} -A "hydrogens" -o {receptor_pdbqt} >/dev/null 2>&1')
        os.system(f'obabel {receptor_pdb} -xr -O {receptor_pdbqt}')

    clean_up_list = []
    v = Vina(sf_name='vina',cpu=n_cpu,verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=center, box_size=[20, 20, 20])
    if n_rigid is None:
        if flexible:
            n_rigid = 1
        else:
            n_rigid = 5
    if exhaustiveness is None:
        if flexible:
            exhaustiveness = 1
        else:
            exhaustiveness = 8
    if flexible:
        os.chdir(output_dir)
        assert receptor_pdb is not None, 'receptor_pdb should be provided when flexible=True'
        os.system(f'{PY27} {CLP} {receptor_pdb} A >/dev/null 2>&1')
        protein_name = os.path.basename(receptor_pdb).replace('.pdb','')
        clean_up_list.append(os.path.join(output_dir,f'{protein_name}_A.pdb'))
        clean_up_list.append(os.path.join(output_dir,f'{protein_name}_A.fasta'))

    ligand_pdbqt_list = sdf2pdbqt(ligand_sdf_list)
    if lig_name_list is None:
        lig_name_list = [f'LIG{i}' for i in range(len(ligand_sdf_list))]
    elif len(lig_name_list) == 1:
        lig_name_list = [f'{lig_name_list[0]}{i}' for i in range(len(ligand_sdf_list))]
    elif len(lig_name_list) != len(ligand_sdf_list):
        raise ValueError('The length of lig_name_list should be 1 or equal to the length of ligand_sdf_list')   
    ligpose_name = []
    ligpose_pdbqt = []
    pose_engery = [[],[]] # rigid, flexible
    #rigid docking
    if verbose: print('rigid docking...')
    for name,ligand_pdbqt in tqdm(list(zip(lig_name_list,ligand_pdbqt_list)), disable=not verbose):
        try:
            v.set_ligand_from_string(ligand_pdbqt)
            v.dock(exhaustiveness=exhaustiveness)
            ridig_poses = v.poses(n_poses=n_rigid)
            ridig_energy = v.energies(n_poses=n_rigid)[:,0].tolist()
            pose_name = [f'{name}_{i}' for i in range(n_rigid)]
            ligpose_name += pose_name
            ligpose_pdbqt += ridig_poses.split('ENDMDL\n')[:-1]
            pose_engery[0] += ridig_energy
        except:
            print(f'Error in docking ligand {name}')
    # ligpose_pdb = pdbqt2pdb(ligpose_pdbqt)
    # for name,pose in zip(ligpose_name,ligpose_pdb):
    #     with open(os.path.join(output_dir,os.path.basename(receptor_pdbqt).replace('.pdbqt',f'_{name}.pdb')), 'w') as f:
    #         f.write(pose.replace('UNL  ','LIG X'))
    ligpose_sdf = pdbqt2sdf(ligpose_pdbqt)
    for name,pose in zip(ligpose_name,ligpose_sdf):
        with open(os.path.join(output_dir,os.path.basename(receptor_pdbqt).replace('.pdbqt',f'_{name}.sdf')), 'w') as f:
            f.write(pose)

    if flexible:
        #flexible docking
        if verbose: print('flexible docking...')
        os.chdir(output_dir)
        pool = mp.Pool(min(n_cpu,len(ligpose_name)))
        tbar = tqdm(total=len(ligpose_name))
        def update(s):
            pose_engery[1].append(s)
            tbar.update()
        for name in ligpose_name:
            pool.apply_async(prep_dock_score,args=(output_dir,name,protein_name,n_flexible),callback=update)
        pool.close()
        pool.join()
           
    for f in clean_up_list:
        if os.path.exists(f):
            os.remove(f)
    with open(os.path.join(output_dir,'ligpose.sc'),'a') as f:
        for name,re,fe in zip(ligpose_name,pose_engery[0],pose_engery[1]):
            f.write(f'{name}\t{re}\t{fe}\n')
    return ligpose_name,pose_engery

if  __name__ == '__main__':
    with open('/home/jiayinjun/Drug_The_Whole_Genome/tmp/chembl.sdf','r') as f:
        ligand_sdf_list = f.read().split('$$$$\n')[:-1]
    ligand_sdf_list = [i+'$$$$\n' for i in ligand_sdf_list]    
    docking_pipeline(
        receptor_pdbqt = None,
        ligand_sdf_list = ligand_sdf_list,
        center=[160.46,173.97,147.77],
        output_dir='/drug/tmp',
        lig_name_list=None,
        n_cpu=16,
        exhaustiveness=32,
        n_rigid=3,
        flexible=True,
        receptor_pdb='/home/jiayinjun/Drug_The_Whole_Genome/tmp/7WFR.pdb',
        n_flexible=150,
    )
