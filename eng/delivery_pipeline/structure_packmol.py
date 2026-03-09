import os
import time
import subprocess
import shutil
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, QED


def _eng_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _delivery_root() -> Path:
    return Path(__file__).resolve().parent


def _maybe_unzip_template_dir(template_dir: str | os.PathLike) -> str:
    p = Path(template_dir)
    if p.is_dir():
        return str(p)
    zip_candidate = p.with_suffix('.zip')
    if zip_candidate.exists():
        p.mkdir(parents=True, exist_ok=True)
        import zipfile
        with zipfile.ZipFile(zip_candidate, 'r') as zf:
            zf.extractall(p)
        return str(p)
    alt_zip = p.parent / 'template.zip'
    if alt_zip.exists() and not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        import zipfile
        with zipfile.ZipFile(alt_zip, 'r') as zf:
            zf.extractall(p)
        # 兼容 zip 内部再套一层 template/
        inner = p / 'template'
        if inner.is_dir() and any(inner.iterdir()):
            for child in inner.iterdir():
                target = p / child.name
                if not target.exists():
                    child.rename(target)
            try:
                inner.rmdir()
            except Exception:
                pass
        return str(p)
    return str(p)


def _candidate_component_dirs(component_dir: str | None = None) -> list[Path]:
    dirs = []
    if component_dir:
        p = Path(component_dir)
        if not p.is_absolute():
            p = _eng_root() / p
        dirs.append(Path(_maybe_unzip_template_dir(p)))
    dirs.extend([
        Path(_maybe_unzip_template_dir(_delivery_root() / 'template')),
        Path(_maybe_unzip_template_dir(_eng_root() / 'LNP' / 'templates')),
    ])
    seen = []
    out = []
    for d in dirs:
        s = str(d.resolve()) if d.exists() else str(d)
        if s not in seen:
            seen.append(s)
            out.append(d)
    return out


def _resolve_component_dir(component_dir: str | None = None) -> str:
    for d in _candidate_component_dirs(component_dir):
        if d.is_dir():
            return str(d.resolve())
    # 返回第一个候选，让后面报错信息更明确
    cand = _candidate_component_dirs(component_dir)
    return str(cand[0]) if cand else str((_delivery_root() / 'template').resolve())


def _find_material_spec(material: str, component_dir: str | None = None):
    for d in _candidate_component_dirs(component_dir):
        spec = d / f'{material}_packmol.json'
        shell = d / f'{material}_shell.pdb'
        if spec.exists() or shell.exists():
            return d, spec, shell
    d = Path(_resolve_component_dir(component_dir))
    return d, d / f'{material}_packmol.json', d / f'{material}_shell.pdb'


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _smiles_to_pdb(smiles: str, out_pdb: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    Chem.MolToPDBFile(mol, out_pdb)
    return out_pdb


def _calc_descriptors_from_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "tPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "RotB": float(Lipinski.NumRotatableBonds(mol)),
        "QED": float(QED.qed(mol)),
    }


def _extract_admet(candidate: dict, fallback_qed: float = 0.0) -> dict:
    ad = candidate.get("admet")
    if isinstance(ad, dict) and ad:
        ad.setdefault("QED", float(ad.get("QED", fallback_qed)))
        return ad
    return {
        "BBB": float(candidate.get("BBB", 0.0)),
        "hERG": float(candidate.get("hERG", 0.0)),
        "AMES": float(candidate.get("AMES", 0.0)),
        "Caco2": float(candidate.get("Caco2", 0.0)),
        "QED": float(candidate.get("QED", fallback_qed)),
    }


def pick_material_and_strategy(descriptors: dict, admet: dict):
    import math
    from pathlib import Path

    def _f(x, default=None):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _s(x, default=""):
        try:
            if x is None:
                return default
            return str(x).strip()
        except Exception:
            return default

    def _in_any(s: str, keywords: list[str]) -> bool:
        s = (s or "").lower()
        return any(k.lower() in s for k in keywords)

    MW = _f(descriptors.get("MW"), 0.0)
    logP = _f(descriptors.get("logP"), None)

    cargo_type = _s(admet.get("cargo_type", descriptors.get("cargo_type", "")), "")
    molecule_type = _s(admet.get("molecule_type", descriptors.get("molecule_type", "")), "")

    is_nucleic = bool(admet.get("is_nucleic_acid", False) or descriptors.get("is_nucleic_acid", False))
    acid_base = _s(admet.get("acid_base", ""), "").lower()

    pKa_main = _f(admet.get("pKa"), None)
    pKa_base = _f(admet.get("pKa_base", None), None)
    pKa_acid = _f(admet.get("pKa_acid", None), None)

    logD74 = _f(admet.get("logD74", None), None)
    logD7 = _f(admet.get("logD7", admet.get("logD_pH7", None)), None)

    sol_raw = admet.get("Solubility", admet.get("solubility", admet.get("logS", None)))
    sol = _f(sol_raw, None)

    if is_nucleic or _in_any(cargo_type, ["nucleic", "sirna", "mrna", "aso", "oligo", "dna", "rna"]) or _in_any(
        molecule_type, ["nucleic", "sirna", "mrna", "aso", "oligo", "dna", "rna"]
    ):
        material_recommended = "LNP"
        strategy = "ionizable_lipid_complexation"
        drug_class = "STEP1_NUCLEIC"
        reason = ["核酸/寡核苷酸 -> 需要离子化脂质复合与内吞逃逸 -> 选 LNP"]
    else:
        reason = []
        strategy = "passive_encapsulation"
        drug_class = "NON_NUCLEIC"
        material_recommended = "PLGA"

    def _logd_weak_base(logP_val: float, pKa_val: float, pH: float) -> float:
        return float(logP_val - math.log10(1.0 + (10.0 ** (pKa_val - pH))))

    def _logd_weak_acid(logP_val: float, pKa_val: float, pH: float) -> float:
        return float(logP_val - math.log10(1.0 + (10.0 ** (pH - pKa_val))))

    def _infer_acid_base(pKa_val):
        if pKa_val is None:
            return ""
        return "base" if pKa_val > 7.0 else "acid"

    pKa_used = None
    pKa_mode = ""

    if acid_base in ("acid", "base"):
        pKa_mode = acid_base
        pKa_used = pKa_main if pKa_main is not None else (pKa_acid if acid_base == "acid" else pKa_base)
    elif acid_base == "zwitterion":
        if (pKa_acid is not None) and (pKa_base is not None) and (logP is not None):
            ld_acid = _logd_weak_acid(logP, pKa_acid, 7.0)
            ld_base = _logd_weak_base(logP, pKa_base, 7.0)
            if ld_base >= ld_acid:
                pKa_mode, pKa_used = "base", pKa_base
            else:
                pKa_mode, pKa_used = "acid", pKa_acid
        else:
            pKa_used = pKa_main
            pKa_mode = _infer_acid_base(pKa_used)
    else:
        if pKa_base is not None:
            pKa_mode, pKa_used = "base", pKa_base
        elif pKa_acid is not None:
            pKa_mode, pKa_used = "acid", pKa_acid
        else:
            pKa_used = pKa_main
            pKa_mode = _infer_acid_base(pKa_used)

    if logP is not None and pKa_used is not None and pKa_mode in ("acid", "base"):
        if logD7 is None:
            logD7 = _logd_weak_acid(logP, pKa_used, 7.0) if pKa_mode == "acid" else _logd_weak_base(logP, pKa_used, 7.0)
        if logD74 is None:
            logD74 = _logd_weak_acid(logP, pKa_used, 7.4) if pKa_mode == "acid" else _logd_weak_base(logP, pKa_used, 7.4)

    sol_ug_ml = None
    if sol is not None:
        if sol <= 0.0:
            S_mol_L = 10.0 ** sol
            g_L = S_mol_L * float(MW if MW else 0.0)
            sol_ug_ml = g_L * 1000.0
        else:
            sol_ug_ml = sol

    is_peptide_like = _in_any(cargo_type, ["protein", "antibody", "peptide"]) or _in_any(molecule_type, ["protein", "antibody", "peptide"])
    if (not is_nucleic) and (not is_peptide_like) and MW and MW < 700.0:
        remote_ok = False
        if (logD7 is not None) and (-2.5 <= logD7 <= 2.0) and (pKa_used is not None) and (pKa_mode in ("acid", "base")):
            if pKa_mode == "base" and pKa_used <= 11.0:
                remote_ok = True
            if pKa_mode == "acid" and pKa_used > 3.0:
                remote_ok = True
        if remote_ok:
            material_recommended = "LIPOSOME"
            strategy = "remote_loading"
            drug_class = "STEP2_REMOTE_LOADING"
            reason.append(f"满足 remote loading 窗口：logD@pH7={logD7:.2f}∈[-2.5,2.0]，且 {pKa_mode} pKa={pKa_used:.2f} 符合阈值 -> 选 PEG-脂质体(remote loading)")

    if (not is_nucleic) and (drug_class not in ("STEP2_REMOTE_LOADING",)):
        if is_peptide_like or (MW and MW >= 700.0):
            material_recommended = "PLGA"
            strategy = "polymer_nanoparticle"
            drug_class = "STEP3A_LARGE_OR_BIOMACRO"
            if is_peptide_like:
                reason.append("蛋白/多肽类型 -> 不走 remote loading/脂质锁药叙事 -> 选 PLGA-PEG(此处用 PLGA 模板)")
            else:
                reason.append(f"MW≈{MW:.1f}≥700Da(工程阈值) -> 选 PLGA-PEG(此处用 PLGA 模板)")
        else:
            prefer_nlc = False
            if logD74 is not None and logD74 >= 2.0:
                prefer_nlc = True
                reason.append(f"logD@7.4={logD74:.2f}≥2 -> 偏好 NLC（疏水小分子锁进脂质基质）")
            elif sol_ug_ml is not None and sol_ug_ml < 10.0:
                prefer_nlc = True
                reason.append(f"溶解度≈{sol_ug_ml:.3g} ug/mL <10 -> 偏好 NLC（难溶药脂质基质锁药）")

            if prefer_nlc:
                material_recommended = "NLC"
                strategy = "lipid_matrix_encapsulation"
                drug_class = "STEP3B_NLC_PREFERRED"
            else:
                material_recommended = "PLGA"
                strategy = "polymer_nanoparticle"
                drug_class = "STEP3B_PLGA_DEFAULT"
                reason.append("未满足 NLC 偏好条件 -> 默认 PLGA-PEG(此处用 PLGA 模板)")

    size_map = {"LNP": 70.0, "LIPOSOME": 90.0, "NLC": 80.0, "PLGA": 100.0}
    size_nm = float(size_map.get(material_recommended, 85.0))
    zeta_mv = -5.0
    drug_loading = 0.12

    component_dir, spec_needed, shell_needed = _find_material_spec(material_recommended)

    material_used = material_recommended
    fallback = None
    if (not shell_needed.exists()) and (not spec_needed.exists()):
        material_used = "PLGA"
        fallback = f"缺少模板 {shell_needed.name} 或 {spec_needed.name}，已回退 PLGA"
        reason.append(fallback)

    return {
        "material": material_used,
        "material_recommended": material_recommended,
        "material_fallback": fallback,
        "drug_class": drug_class,
        "strategy": strategy,
        "decision_features": {
            "cargo_type": cargo_type,
            "molecule_type": molecule_type,
            "MW": MW,
            "logP": logP,
            "pKa_used": pKa_used,
            "pKa_mode": pKa_mode,
            "logD7": logD7,
            "logD74": logD74,
            "Solubility_raw": sol_raw,
            "Solubility_ug_mL_est": sol_ug_ml,
        },
        "decision_reason": "; ".join([r for r in reason if r]),
        "targeting_ligand": None,
        "bbb_method": "none",
        "size_nm": size_nm,
        "zeta_mv": zeta_mv,
        "drug_loading": drug_loading,
    }


def enumerate_designs(candidate: dict, out_root: str, n_each: int = 1, **kwargs):
    cand_id = candidate.get("candidate_id", "CAND")
    smiles = candidate.get("smiles")
    desc = candidate.get("descriptors")
    if not isinstance(desc, dict) or not desc:
        desc = _calc_descriptors_from_smiles(smiles)
    admet = _extract_admet(candidate, fallback_qed=float(desc.get("QED", 0.0)))
    base = pick_material_and_strategy(desc, admet)
    n = max(1, int(n_each))
    designs = []
    for i in range(n):
        dtag = f"D{(i+1):03d}"
        workdir = os.path.join(out_root, cand_id, dtag)
        designs.append({
            "design_id": f"{cand_id}_{dtag}",
            "candidate_id": cand_id,
            "smiles": smiles,
            "descriptors": desc,
            "admet": admet,
            "material": base["material"],
            "strategy": base.get("strategy", "passive_encapsulation"),
            "targeting_ligand": base["targeting_ligand"],
            "bbb_method": base["bbb_method"],
            "size_nm": float(base["size_nm"]),
            "zeta_mv": float(base["zeta_mv"]),
            "drug_loading": float(base["drug_loading"]),
            "workdir": workdir,
        })
    return designs


def _copy_to_scratch(src_path: str, scratch_dir: str, rename_to: str | None = None) -> str:
    dst = os.path.join(scratch_dir, rename_to or os.path.basename(src_path))
    shutil.copy2(src_path, dst)
    return dst


def _normalize_atom_ids(atom_ids):
    out = []
    for x in (atom_ids or []):
        try:
            out.append(int(x))
        except Exception:
            pass
    return sorted(set([i for i in out if i >= 1]))


def _packmol_atoms_block(atom_ids, region_line: str) -> list[str]:
    atom_ids = _normalize_atom_ids(atom_ids)
    if not atom_ids:
        return []
    lines = ["  atoms " + " ".join(map(str, atom_ids)), f"    {region_line}", "  end atoms"]
    return lines


def _build_advanced_packmol_input_from_spec(spec: dict, output_pdb: str, size_nm: float) -> str:
    mode = str(spec.get("mode", "matrix_shell")).strip().lower()
    center = spec.get("center", [0.0, 0.0, 0.0])
    cx, cy, cz = [float(v) for v in (center + [0, 0, 0])[:3]]
    tol = float(spec.get("tolerance", 2.0))
    seed = int(spec.get("seed", 12345))
    shell_thickness = float(spec.get("shell_thickness_A", 20.0))
    core_radius = float(spec.get("core_radius_A", max(35.0, size_nm * 3.0)))
    outer_radius = float(spec.get("outer_radius_A", max(core_radius + shell_thickness, size_nm * 5.0)))
    inner_leaflet_mid = float(spec.get("inner_leaflet_mid_A", max(10.0, core_radius - shell_thickness * 0.5)))
    outer_leaflet_mid = float(spec.get("outer_leaflet_mid_A", core_radius + shell_thickness * 0.5))
    drug_region = str(spec.get("drug_region", "core")).lower()
    drug_count = int(spec.get("drug_count", 40))

    lines = [
        f"tolerance {tol:.3f}",
        "filetype pdb",
        f"output {output_pdb}",
        f"seed {seed}",
        "",
    ]

    components = spec.get("components", [])
    if not isinstance(components, list) or not components:
        raise ValueError("高级packmol模式需要 components 列表")

    def shell_region(r_in: float, r_out: float) -> list[str]:
        return [
            f"  inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {r_out:.3f}",
            f"  outside sphere {cx:.3f} {cy:.3f} {cz:.3f} {r_in:.3f}",
        ]

    for comp in components:
        pdb_name = str(comp["pdb"]).strip()
        role = str(comp.get("role", "shell")).strip().lower()
        count = int(comp.get("count", 0))
        if count <= 0:
            continue
        head_atoms = _normalize_atom_ids(comp.get("head_atom_ids", []))
        tail_atoms = _normalize_atom_ids(comp.get("tail_atom_ids", []))
        block = [f"structure {pdb_name}", f"  number {count}"]

        if mode == "vesicle_bilayer" and role in ("outer_leaflet", "inner_leaflet"):
            if role == "outer_leaflet":
                leaflet_r_in = float(comp.get("r_in_A", outer_leaflet_mid - shell_thickness * 0.5))
                leaflet_r_out = float(comp.get("r_out_A", outer_leaflet_mid + shell_thickness * 0.5))
                block += shell_region(leaflet_r_in, leaflet_r_out)
                if head_atoms:
                    block += _packmol_atoms_block(head_atoms, f"outside sphere {cx:.3f} {cy:.3f} {cz:.3f} {outer_leaflet_mid:.3f}")
                if tail_atoms:
                    block += _packmol_atoms_block(tail_atoms, f"inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {outer_leaflet_mid:.3f}")
            else:
                leaflet_r_in = float(comp.get("r_in_A", inner_leaflet_mid - shell_thickness * 0.5))
                leaflet_r_out = float(comp.get("r_out_A", inner_leaflet_mid + shell_thickness * 0.5))
                block += shell_region(leaflet_r_in, leaflet_r_out)
                if head_atoms:
                    block += _packmol_atoms_block(head_atoms, f"inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {inner_leaflet_mid:.3f}")
                if tail_atoms:
                    block += _packmol_atoms_block(tail_atoms, f"outside sphere {cx:.3f} {cy:.3f} {cz:.3f} {inner_leaflet_mid:.3f}")
        elif role == "core":
            block += [f"  inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {core_radius:.3f}"]
        elif role == "shell":
            block += shell_region(core_radius, outer_radius)
        else:
            r_in = float(comp.get("r_in_A", core_radius))
            r_out = float(comp.get("r_out_A", outer_radius))
            block += shell_region(r_in, r_out)

        for ax in ("x", "y", "z"):
            if f"constrain_rotation_{ax}" in comp:
                ang, wiggle = comp[f"constrain_rotation_{ax}"]
                block += [f"  constrain_rotation {ax} {float(ang):.2f} {float(wiggle):.2f}"]

        block += ["end structure", ""]
        lines += block

    if drug_count > 0:
        lines += ["structure drug.pdb", f"  number {drug_count}"]
        if drug_region == "core":
            lines += [f"  inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {core_radius:.3f}"]
        elif drug_region == "shell":
            lines += shell_region(core_radius, outer_radius)
        elif drug_region == "inner_aqueous":
            lines += [f"  inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {inner_leaflet_mid - shell_thickness * 0.5:.3f}"]
        else:
            lines += [f"  inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {core_radius:.3f}"]
        lines += ["end structure", ""]

    return "\n".join(lines)


def _load_material_packmol_spec(component_dir: str, material: str):
    spec_path = os.path.join(component_dir, f"{material}_packmol.json")
    if not os.path.exists(spec_path):
        return None, spec_path
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    return spec, spec_path


def _prepare_advanced_components(spec: dict, component_dir: str, scratch: str):
    copied = []
    for comp in spec.get("components", []):
        pdb_name = str(comp.get("pdb", "")).strip()
        if not pdb_name:
            continue
        src = os.path.join(component_dir, pdb_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"高级packmol缺少组分PDB: {src}")
        _copy_to_scratch(src, scratch, rename_to=os.path.basename(pdb_name))
        copied.append(os.path.basename(pdb_name))
    return copied


def _build_default_simple_packmol_input(output_pdb: str, size_nm: float):
    r_outer = max(80.0, size_nm * 5.0)
    r_inner = max(40.0, r_outer * 0.6)
    return "\n".join([
        "tolerance 2.0",
        "filetype pdb",
        f"output {output_pdb}",
        "seed 12345",
        "",
        "structure PLGA_shell.pdb",
        "  number 180",
        f"  inside sphere 0. 0. 0. {r_outer:.2f}",
        f"  outside sphere 0. 0. 0. {r_inner:.2f}",
        "end structure",
        "",
        "structure drug.pdb",
        "  number 40",
        f"  inside sphere 0. 0. 0. {r_inner:.2f}",
        "end structure",
        "",
    ])


def run_packmol(design: dict, packmol_bin: str = "packmol", component_dir: str = "delivery_pipeline/template", **kwargs):
    debug_on = True
    orig_workdir = os.path.abspath(design["workdir"])
    os.makedirs(orig_workdir, exist_ok=True)

    packmol_bin = (os.environ.get("PACKMOL_BIN", packmol_bin) or "packmol").strip().strip('\"')
    component_dir = _resolve_component_dir(component_dir)

    tmp_root = os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
    scratch = os.path.join(tmp_root, "packmol_scratch", design.get("candidate_id", "CAND"), design.get("design_id", "D001"))
    os.makedirs(scratch, exist_ok=True)

    drug_pdb_s = os.path.join(scratch, "drug.pdb")
    inp_file_s = os.path.join(scratch, "packmol.inp")
    out_pdb_s = os.path.join(scratch, "system_packmol.pdb")
    log_file_s = os.path.join(scratch, "packmol.log")

    out_pdb_o = os.path.join(orig_workdir, "system_packmol.pdb")
    log_file_o = os.path.join(orig_workdir, "packmol.log")
    inp_file_o = os.path.join(orig_workdir, "packmol.inp")

    design["packmol_debug"] = {
        "debug_on": debug_on,
        "module_file": __file__,
        "orig_workdir": orig_workdir,
        "scratch": scratch,
        "packmol_bin": packmol_bin,
        "component_dir": component_dir,
        "component_dir_isdir": os.path.isdir(component_dir),
        "which_packmol_bin": shutil.which(packmol_bin),
        "PATH_head": os.environ.get("PATH", "")[:400],
    }

    try:
        try:
            design["packmol_debug"]["component_dir_listing"] = sorted(os.listdir(component_dir))
        except Exception as e:
            design["packmol_debug"]["component_dir_listing_error"] = repr(e)

        _smiles_to_pdb(design["smiles"], drug_pdb_s)
        material = str(design.get("material", "PLGA") or "PLGA").strip()
        design["packmol_debug"]["material_used"] = material

        spec, spec_path = _load_material_packmol_spec(component_dir, material)
        design["packmol_debug"]["spec_path"] = spec_path
        design["packmol_debug"]["spec_exists"] = bool(spec)

        for _p in [out_pdb_s, out_pdb_o, log_file_s, log_file_o, inp_file_o]:
            try:
                if os.path.exists(_p):
                    os.remove(_p)
            except Exception:
                pass

        size_nm = float(design.get("size_nm", 85.0))
        if spec:
            copied_components = _prepare_advanced_components(spec, component_dir, scratch)
            design["packmol_debug"]["advanced_components"] = copied_components
            inp = _build_advanced_packmol_input_from_spec(spec, "system_packmol.pdb", size_nm=size_nm)
            design["packmol_debug"]["packmol_mode"] = str(spec.get("mode", "matrix_shell"))
        else:
            raise FileNotFoundError(
                f"未找到高级packmol规格文件：{spec_path}\n"
                f"当前已禁用简单模式，不再接受 {material}_shell.pdb 回退。\n"
                f"请在 {component_dir} 中提供 {material}_packmol.json 及其依赖的真实组分PDB。"
            )

        Path(inp_file_s).write_text(inp, encoding="utf-8")
        shutil.copy2(inp_file_s, inp_file_o)
        design["packmol_debug"]["packmol_inp_saved_to"] = inp_file_o
        design["packmol_debug"]["inp_head"] = inp[:2000]

        # =========================
        # 关键修复：支持两类 PACKMOL_BIN
        # 1) 绝对/相对路径，指向 exe / bat / cmd
        # 2) 仅给一个命令名，例如 packmol，走 PATH
        # =========================
        packmol_bin = str(packmol_bin).strip().strip('"').strip("'")
        resolved_packmol = packmol_bin
        if os.path.sep in packmol_bin or ('/' in packmol_bin) or ('\\' in packmol_bin):
            if not os.path.exists(packmol_bin):
                raise FileNotFoundError(f"PACKMOL_BIN 不存在: {packmol_bin}")
        else:
            which_packmol = shutil.which(packmol_bin)
            if which_packmol:
                resolved_packmol = which_packmol

        if os.name == "nt" and resolved_packmol.lower().endswith((".bat", ".cmd")):
            cmd_to_run = ["cmd", "/d", "/s", "/c", "call", resolved_packmol]
        else:
            cmd_to_run = [resolved_packmol]

        design["packmol_debug"]["resolved_packmol"] = resolved_packmol

        design["packmol_debug"]["cmd_to_run"] = cmd_to_run

        timeout_sec = int(os.environ.get("PACKMOL_TIMEOUT_SEC", "1800"))
        design["packmol_debug"]["timeout_sec"] = timeout_sec

        with open(inp_file_s, "r", encoding="utf-8") as fin, open(log_file_s, "w", encoding="utf-8") as flog:
            flog.write(f"[DEBUG] scratch={scratch}\n")
            flog.write(f"[DEBUG] packmol_bin={repr(packmol_bin)}\n")
            flog.write(f"[DEBUG] cmd_to_run={cmd_to_run}\n")
            flog.write(f"[DEBUG] inp_file={inp_file_s}\n\n")
            flog.flush()

            proc = subprocess.Popen(
                cmd_to_run,
                stdin=fin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=scratch,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )

            t0 = time.time()
            last_line_ts = t0

            while True:
                line = proc.stdout.readline()
                if line:
                    last_line_ts = time.time()
                    flog.write(line)
                    flog.flush()
                    print("[PACKMOL]", line.rstrip())
                else:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.05)

                if time.time() - t0 > timeout_sec:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    flog.write(f"\n[DEBUG] TIMEOUT after {timeout_sec}s\n")
                    flog.flush()
                    raise RuntimeError(f"PACKMOL 超时：{timeout_sec} 秒仍未结束（可设置环境变量 PACKMOL_TIMEOUT_SEC 调大）")

                if time.time() - last_line_ts > 300:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    flog.write("\n[DEBUG] NO OUTPUT for 300s, killed.\n")
                    flog.flush()
                    raise RuntimeError("PACKMOL 5分钟无任何输出，已终止。请检查 packmol.log / packmol.inp 是否正确。")

            ret = proc.wait()
            flog.write(f"\n[DEBUG] returncode={ret}\n")
            flog.write(f"[DEBUG] out_pdb_exists={os.path.exists(out_pdb_s)}\n")
            flog.flush()

        class _RC:
            def __init__(self, rc):
                self.returncode = rc

        proc = _RC(ret)

        shutil.copy2(log_file_s, log_file_o)
        if os.path.exists(out_pdb_s):
            shutil.copy2(out_pdb_s, out_pdb_o)

        atom_lines = 0
        resnames = set()
        if os.path.exists(out_pdb_o):
            for line in Path(out_pdb_o).read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(("ATOM", "HETATM")):
                    atom_lines += 1
                    rn = line[17:20].strip()
                    if rn:
                        resnames.add(rn)

        design["packmol_debug"]["out_resnames"] = sorted(list(resnames))[:50]
        design["packmol_debug"]["out_atom_lines"] = int(atom_lines)
        design["packmol_ok"] = bool(proc.returncode == 0 and os.path.exists(out_pdb_o) and atom_lines > 0)
        design["packmol_pdb"] = out_pdb_o if os.path.exists(out_pdb_o) else None
        design["packmol_log"] = log_file_o

        try:
            tail = Path(log_file_o).read_text(encoding="utf-8", errors="ignore")[-3000:]
        except Exception as e:
            tail = f"[READ_LOG_ERROR] {repr(e)}"
        design["packmol_debug"]["log_tail"] = tail

        return design

    except Exception as e:
        design["packmol_ok"] = False
        design["packmol_pdb"] = None
        design["packmol_log"] = log_file_o
        design.setdefault("packmol_debug", {})["error"] = repr(e)
        try:
            Path(log_file_o).write_text(f"[EXCEPTION]\n{repr(e)}\n", encoding="utf-8")
        except Exception:
            pass
        return design