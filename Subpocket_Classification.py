import os, math, requests, time
import numpy as np
import pandas as pd
from rdkit import Chem
from collections import Counter

# --- data 읽어오기
def load_region_centroids(region_path):
    region_centroids = {}
    for region in os.listdir(region_path):
        file_path = os.path.join(region_path, region)
        region_name = region[:1].upper()
        with open(file_path) as f:
            lines = f.readlines()
        for line in lines:
            if len(line.split()) == 6:
                parts = line.split()
                if parts[3] == 'N':
                    x, y, z = map(float, parts[:3])
                    region_centroids[region_name] = (x, y, z)
    return region_centroids

def load_frags_for_each_ligand(fragment_path):
    fragment_coords_for_each_ligand = {}
    fragment_bonds_for_each_ligand = {}
    for ligand_folder in os.listdir(fragment_path):
        ligand_path = os.path.join(fragment_path, ligand_folder)
        if not os.path.isdir(ligand_path):
            continue
        ligand_name = ligand_folder.replace("_frags", "")
        frag_dict = {}
        bond_dict = {}

        for frag_file in os.listdir(ligand_path):
            frag_path = os.path.join(ligand_path, frag_file)
            frag_coords = {}
            frag_bonds = {}
            atom_idx = 1
            bond_idx = 1
            lines = open(frag_path).read().splitlines()

            for line in lines:
                if len(line.split()) == 16:
                    parts = line.split()
                    x, y, z = map(float, parts[:3])
                    atom_type = parts[3]
                    frag_coords[atom_idx] = (x, y, z, atom_type)
                    atom_idx += 1
                if len(line.split()) == 4 and len(line) == 12:
                    parts = line.split()
                    i1 = int(parts[0])
                    i2 = int(parts[1])
                    i3 = int(parts[2])
                    frag_bonds[bond_idx] = (i1, i2, i3)
                    bond_idx += 1
            frag_dict[frag_file] = frag_coords
            bond_dict[frag_file] = frag_bonds
        fragment_coords_for_each_ligand[ligand_name] = frag_dict
        fragment_bonds_for_each_ligand[ligand_name] = bond_dict
        print(f"{ligand_name} coords, bonds information parsed, starting TWN subpocket assigning...")
    return fragment_coords_for_each_ligand, fragment_bonds_for_each_ligand

def TWN_subpocket_based_Classification(region_dict, fragment_coords_for_each_ligand):
    for ligand_name, fragments in fragment_coords_for_each_ligand.items():
        for frag_file, atom_dict in fragments.items():
            new_frag_dict = {}
            nearest_key = None
            for idx, (x, y, z, atom_type) in atom_dict.items():
                min_dist = float('inf')
                for key, (rx, ry, rz) in region_dict.items():
                    dist = math.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        if atom_type == 'R' or atom_type == 'H':  # atom_type이 R과 H일 경우에는 region서 제외
                            nearest_key = ""
                        else:
                            nearest_key = key
                new_frag_dict[idx] = (x, y, z, atom_type, nearest_key)
            fragments[frag_file] = new_frag_dict
            print(f"{frag_file}"'s all atom were assigned its TWN-subpocket')
    return fragment_coords_for_each_ligand

def klifs_parsing(kinase_names, ligand_files):
    base_url = "https://klifs.net/api_v2"
    kinase_info = {}
    for idx, name in enumerate(kinase_names):
        r = requests.get(f"{base_url}/kinase_ID", params={"kinase_name": name, "species": "Human"})
        if r.status_code != 200 or not r.json():
            print(f"[{name}] Kinase ID 조회 실패")
            kinase_info[ligand_files[idx]] = ["NA"] * 7
            continue
        kd = r.json()[0]
        family   = kd.get("family")
        subfam   = kd.get("subfamily")
        group    = kd.get("group")
        kinase_id= kd.get("kinase_ID")

        crystal, ligand_pdb = ligand_files[idx].split('_', 1)
        r2 = requests.get(f"{base_url}/structures_list?kinase_ID={kinase_id}")
        alt = chain = "NA"
        if r2.status_code == 200 and r2.json():
            for entry in r2.json():
                if entry.get("pdb") == crystal:
                    alt   = entry.get("alt","NA")
                    chain = entry.get("chain","NA")
                    break
        kinase_info[ligand_files[idx]] = [name, family, subfam, group, ligand_pdb, alt, chain]
        print(f"{kinase_info[ligand_files[idx]]}'s kinase info were parsed at KLIFS")
    return kinase_info

def define_TWN_subpocket(assigned_dictionary, region_dict, frag_path):
    """
    각 ligand에 대해 fragment 파일별로 가장 대표적인 TWN subpocket region을 반환
    """
    ligand_folder = os.path.basename(os.path.dirname(frag_path))
    ligand_name = ligand_folder.replace("_frags", "")
    frag_file = os.path.basename(frag_path)
    atom_proper_regs = assigned_dictionary[ligand_name][frag_file]
    # 각 atom에서 subpocket 문자만 뽑아서 리스트로
    atom_regs = [atom_proper_regs[i+1][-1] for i in range(len(atom_proper_regs))]
    atom_regs = [r for r in atom_regs if r and r.isalpha()]
    # 가장 많이 등장한 문자 하나 선택
    char_counts = Counter(atom_regs)
    most_common = char_counts.most_common()
    max_count = most_common[0][1]
    # atom-subpocket이 절대 다수인 것이 하나면 바로 선택
    top_regions = [c for c, cnt in most_common if cnt == max_count]
    if len(top_regions) == 1:
        region = top_regions[0]
    else:
        # 동점일 때는 heavy-atom centroid 기반으로 결정
        coords = [
            (x, y, z)
            for x, y, z, atom_type, _ in atom_proper_regs.values()
            if atom_type != ('H', 'R')
        ]
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        min_dist = float('inf')
        region = None
        for key, (rx, ry, rz) in region_dict.items():
            d = math.sqrt((cx - rx) ** 2 + (cy - ry) ** 2 + (cz - rz) ** 2)
            if d < min_dist:
                min_dist = d
                region = key
        print(f"{frag_file}'s proper subpocket is {region}")
    return region

def sdf_rewrite(region_assigned, kinase_info, frag_path, frag_subpocket):
    if not os.path.isfile(frag_path) or not frag_path.endswith(".sdf"):
        print(f"[SKIP] invalid path: {frag_path}")
        return

    ligand_folder = os.path.basename(os.path.dirname(frag_path)) #1bmk_SB5_A_frag_001
    ligand_name   = ligand_folder.replace("_frags", "") #1bmk_SB5_A
    frag_file     = os.path.basename(frag_path) #1bmk_SB5_A_frag_001.sdf

    if ligand_name not in region_assigned or ligand_name not in kinase_info:
        print(f"[SKIP] no data for {ligand_name}")
        return
    if frag_file not in region_assigned[ligand_name]:
        print(f"[SKIP] no region info for {frag_file}")
        return

    content = open(frag_path).read()
    kin_meta = kinase_info[ligand_name]
    region_info = region_assigned[ligand_name][frag_file]
    block = []
    block.append(">  <kinase>")
    block.append(kin_meta[0])
    block.append("")
    block.append(">  <family>")
    block.append(kin_meta[1])
    block.append("")
    block.append(">  <subfamily>")
    block.append(kin_meta[2])
    block.append("")
    block.append(">  <group>")
    block.append(kin_meta[3])
    block.append("")
    block.append(">  <ligand_pdb>")
    block.append(kin_meta[4])
    block.append("")
    block.append(">  <alt>")
    block.append(kin_meta[5])
    block.append("")
    block.append(">  <chain>")
    block.append(kin_meta[6])
    block.append("")
    block.append(">  <atom.prop.subpocket>")
    regs = [ region_info[i+1][-1] for i in range(len(region_info)) ]
    regs = [r for r in regs if r and r.isalpha()]
    block.append(" ".join(regs))
    block.append("")
    block.append(">  <frag.prop.subpocket>")
    block.append(frag_subpocket)
    block.append("")

    if content.strip().endswith("$$$$"):
        body = content.rsplit("$$$$", 1)[0]
        # 이미 <atom.prop.subpocket> 블록이 들어있지 않을 때만 block을 추가
        if not any(line.startswith(">  <atom.prop.subpocket>") for line in body.splitlines()):
            new_content = body.rstrip() + "\n" + "\n".join(block) + "\n$$$$\n"
        else:
            new_content = content
    else:
        new_content = content
    with open(frag_path, 'w') as w:
        w.write(new_content)
    print(f"[UPDATED] {frag_file}")

# === 사용 예시 ===
if __name__ == "__main__":
    start = time.time()

    region_path = "./regions"
    fragment_sdf_path = "./KLIFS_ligand_fragments"
    df = pd.read_excel("./ligands(filtered_final).xlsx")
    kinase_names = df["Target"].tolist()
    ligand_files = df["file"].tolist()

    region_centroids   = load_region_centroids(region_path)
    frags_each_ligand, bonds_each_ligand  = load_frags_for_each_ligand(fragment_sdf_path)
    assigned           = TWN_subpocket_based_Classification(region_centroids, frags_each_ligand)
    kinase_info        = klifs_parsing(kinase_names, ligand_files)

    frag_TWN = {}
    for ligand_folder in os.listdir(fragment_sdf_path):
        ligand_dir = os.path.join(fragment_sdf_path, ligand_folder)
        if not os.path.isdir(ligand_dir):
            continue
        for frag in os.listdir(ligand_dir):
            print(f"[INFO] processing {frag}")
            if frag.endswith(".sdf"):
                region = define_TWN_subpocket(assigned, region_centroids, os.path.join(ligand_dir,frag))
                sdf_rewrite(assigned, kinase_info, os.path.join(ligand_dir, frag), region)

    end = time.time()
    print(f"\nrun time: {end - start:.2f}sec")