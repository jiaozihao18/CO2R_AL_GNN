import pickle
import os
import lzma
import multiprocessing as mp
from tqdm import tqdm
import ase.io
from ase.io import extxyz
import shutil
from elements import elements
import re

def download(i):
    
    """
    i--i--i--randomxxx.extxyz.xz
        --system.txt
     --i.tar
    """

    if not os.path.isfile(f"./{i}/{i}.tar"):
        os.makedirs(f"./{i}", exist_ok=True)
        download_link = f"https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/{i}.tar"
        os.system(f"wget {download_link} -P {i}")
        
    if not os.path.isdir(f"./{i}/{i}"):
        os.system(f"tar -xf {i}/{i}.tar -C {i}")

# convert extxyz.xz to extxyz
def decompress_xz(ip_op_pair):
    ip_file, op_file = ip_op_pair
    with lzma.open(ip_file, "rb") as ip:
        with open(op_file, "wb") as op:
            shutil.copyfileobj(ip, op)

def screen(txt_path, data_mapping):
    sid_list = []
    ref_ene_list = []
    ts_ele = set([ele.Symbol for ele in elements.Transition_Metals])
    with open(txt_path, 'r') as fd:
        for line in fd:
            sid, ref_ene = line.strip().split(sep=',')
            mapping_dict = data_mapping[sid]
            sid_ele = re.sub(r"\d+", "", mapping_dict["bulk_symbols"])
            sid_ele = set(re.findall('[A-Z][^A-Z]*', sid_ele))
            # transition metals
            if mapping_dict['anomaly'] == 0 and sid_ele < ts_ele:
                sid_list.append(sid)
                ref_ene_list.append(ref_ene)
                
    return sid_list, ref_ene_list

def main(data_mapping, i):
    
    sid_list, ref_ene_list = screen(f"./{i}/{i}/system.txt", data_mapping)
    
    if len(sid_list) > 0:
        
        # decompressing selected extxyz.xz file
        os.makedirs(f"./{i}/{i}/temp", exist_ok=True)

        ip_op_pairs = []
        for sid in sid_list:
            ip_file = f"./{i}/{i}/{i}/{sid}.extxyz.xz"
            op_file = f"./{i}/{i}/temp/{sid}.extxyz"
            ip_op_pairs.append((ip_file, op_file))

        pool = mp.Pool(4)
        list(
            tqdm(
                pool.imap(decompress_xz, ip_op_pairs),
                total=len(ip_op_pairs),
                desc="Decompressing %s" % i,
            )
        )

        atoms_list = []
        for sid in tqdm(sid_list, desc="Extracting atoms"):
            # extract the last frame
            atoms = ase.io.read(f"./{i}/{i}/temp/{sid}.extxyz", index=-1)
            atoms_list.append(atoms)

        # write new extxyz file
        os.makedirs("./target_dir", exist_ok=True)
        
        columns = (['symbols', 'positions', 'move_mask', 'tags'])
        with open(f"./target_dir/{i}.extxyz", 'w') as f:
            for atoms in atoms_list:
                extxyz.write_xyz(f, atoms, columns=columns, append=True)

        # write txt file 
        with open(f"./target_dir/{i}.txt", 'w') as f:
            for sid, ref_ene in zip(sid_list, ref_ene_list):
                f.write("%s,frame-1,%s\n" %(sid, ref_ene))
        
    os.system(f"rm -r {i}")
    

if __name__ == '__main__':
    
    data_mapping_path='./oc20_data_mapping.pkl'
    f = open(data_mapping_path, 'rb')
    data_mapping = pickle.load(f)
    
    for i in range(4, 57):
        if i not in [5, 8, 36, 37, 43]:
            print('*'*25+f'{i}'+'*'*25)
            download(i=i)
            main(data_mapping=data_mapping, i=i)