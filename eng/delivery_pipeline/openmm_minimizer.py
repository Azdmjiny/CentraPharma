import os
import math
import json
from typing import Optional, Sequence

from openmm.app import (
    PDBFile,
    Simulation,
    PME,
    NoCutoff,
    HBonds,
    AmberPrmtopFile,
    AmberInpcrdFile,
)
from openmm import Platform, XmlSerializer, LangevinMiddleIntegrator
from openmm import unit


def _ensure_dir(p):
    d = os.path.dirname(os.path.abspath(p))
    if d:
        os.makedirs(d, exist_ok=True)


def _vec3_to_A(v):
    return (float(v.x) * 10.0, float(v.y) * 10.0, float(v.z) * 10.0)  # nm -> Å


def _mean_xyz(points):
    n = len(points)
    sx = sy = sz = 0.0
    for x, y, z in points:
        sx += x
        sy += y
        sz += z
    return (sx / n, sy / n, sz / n)


def _norm3(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _mean_value(values):
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _pick_platform(preferred: Optional[Sequence[str]] = None):
    if preferred is None:
        preferred = ["CPU", "CUDA", "OpenCL"]

    available = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]

    errors = {}

    for name in preferred:
        if name not in available:
            continue

        try:
            platform = Platform.getPlatformByName(name)
            properties = {}
            if name in ("CUDA", "OpenCL"):
                properties["Precision"] = "mixed"
            return platform, properties, available
        except Exception as e:
            errors[name] = str(e)
            continue

    raise RuntimeError(f"没有可用的平台。可用平台: {available}；尝试失败信息: {errors}")
def compute_packaging_metrics(
    pdb_path: str,
    core_radius_A: float = 125.0,
    drug_chain_id: str = "D",
    atoms_per_drug: int = 57,
):
    """
    计算几个简单包装指标。
    注意：这个函数仍然沿用你旧版的假设：
    - 药物链是 D
    - 每个药物 57 个原子
    - 核心半径 125 Å

    后面如果你切到真正动态组装，这几个参数最好不要再写死。
    """

    pdb = PDBFile(pdb_path)
    atoms = list(pdb.topology.atoms())

    all_positions_A = [_vec3_to_A(p) for p in pdb.positions]
    nanoparticle_center = _mean_xyz(all_positions_A)

    drug_positions_A = []
    for atom, pos in zip(atoms, all_positions_A):
        if atom.residue.chain.id == drug_chain_id:
            drug_positions_A.append(pos)

    if not drug_positions_A:
        raise RuntimeError(f"在 PDB 中没有找到药物链 '{drug_chain_id}'，无法计算包装指标。")

    if len(drug_positions_A) % atoms_per_drug != 0:
        raise RuntimeError(
            f"药物链 '{drug_chain_id}' 原子数 = {len(drug_positions_A)}，"
            f"不能被 atoms_per_drug = {atoms_per_drug} 整除。"
        )

    n_drugs = len(drug_positions_A) // atoms_per_drug

    drug_centers = []
    boundary_margins = []
    inside_count = 0

    for i in range(n_drugs):
        block = drug_positions_A[i * atoms_per_drug:(i + 1) * atoms_per_drug]

        c = _mean_xyz(block)
        drug_centers.append(c)

        r_max = max(_norm3(p, nanoparticle_center) for p in block)
        delta_i = float(core_radius_A - r_max)
        boundary_margins.append(delta_i)

        if delta_i > 0.0:
            inside_count += 1

    f_in = float(inside_count / n_drugs)
    delta_min_A = float(min(boundary_margins))
    drug_radius_A = [float(_norm3(c, nanoparticle_center)) for c in drug_centers]

    drug_nn_dist_A = []
    for i, ci in enumerate(drug_centers):
        dmin = None
        for j, cj in enumerate(drug_centers):
            if i == j:
                continue
            d = _norm3(ci, cj)
            if dmin is None or d < dmin:
                dmin = d
        drug_nn_dist_A.append(float(dmin if dmin is not None else 0.0))

    return {
        "f_in": f_in,
        "delta_min_A": delta_min_A,
        "drug_radius_mean_A": _mean_value(drug_radius_A),
        "drug_nn_dist_mean_A": _mean_value(drug_nn_dist_A),
    }


def run_openmm_minimization(
    output_pdb: str,
    prmtop_path: str,
    inpcrd_path: str,
    state_xml_path: Optional[str] = None,
    summary_json_path: Optional[str] = None,
    max_iterations: int = 5000,
    nonbonded_cutoff_nm: float = 1.0,
    constraints: str = "HBonds",
    temperature_K: float = 300.0,
    friction_per_ps: float = 1.0,
    timestep_fs: float = 2.0,
    platform_preference: Optional[Sequence[str]] = None,
    verbose: bool = False,
):
    """
    真实 OpenMM 能量最小化入口。

    需要 AmberTools 先生成：
    - prmtop
    - inpcrd

    参数说明：
    - output_pdb: 最小化后输出的 PDB
    - prmtop_path: Amber 拓扑文件
    - inpcrd_path: Amber 坐标文件
    - state_xml_path: 可选，保存 OpenMM state.xml
    - summary_json_path: 可选，保存最小化摘要 JSON
    - max_iterations: 最大最小化步数
    """

    if not prmtop_path or not os.path.exists(prmtop_path):
        raise FileNotFoundError(f"找不到 prmtop 文件: {prmtop_path}")
    if not inpcrd_path or not os.path.exists(inpcrd_path):
        raise FileNotFoundError(f"找不到 inpcrd 文件: {inpcrd_path}")

    _ensure_dir(output_pdb)
    if state_xml_path:
        _ensure_dir(state_xml_path)
    if summary_json_path:
        _ensure_dir(summary_json_path)

    prmtop = AmberPrmtopFile(prmtop_path)
    inpcrd = AmberInpcrdFile(inpcrd_path)

    constraint_map = {
        "HBonds": HBonds,
        "h-bonds": HBonds,
        "hbonds": HBonds,
        None: HBonds,
    }
    constraint_mode = constraint_map.get(constraints, HBonds)

    box_vectors = inpcrd.boxVectors
    if box_vectors is None:
        try:
            box_vectors = prmtop.topology.getPeriodicBoxVectors()
        except Exception:
            box_vectors = None

    is_periodic = box_vectors is not None
    nonbonded_method = PME if is_periodic else NoCutoff

    system = prmtop.createSystem(
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=float(nonbonded_cutoff_nm) * unit.nanometer,
        constraints=constraint_mode,
    )

    integrator = LangevinMiddleIntegrator(
        float(temperature_K) * unit.kelvin,
        float(friction_per_ps) / unit.picosecond,
        float(timestep_fs) * unit.femtoseconds,
    )

    platform, properties, available_platforms = _pick_platform(platform_preference)

    if verbose:
        print(f"[OPENMM] prmtop={prmtop_path}")
        print(f"[OPENMM] inpcrd={inpcrd_path}")
        print(f"[OPENMM] preferred platform={platform.getName()}")
        print(f"[OPENMM] available={available_platforms}")
        print(f"[OPENMM] periodic={is_periodic}")
        print(f"[OPENMM] nonbondedMethod={'PME' if is_periodic else 'NoCutoff'}")
        print(f"[OPENMM] max_iterations={max_iterations}")

    try:
        simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
        simulation.context.setPositions(inpcrd.positions)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
    except Exception as e:
        if platform.getName() != "CPU" and "CPU" in available_platforms:
            fallback_from = platform.getName()
            fallback_error = repr(e)
            cpu_platform = Platform.getPlatformByName("CPU")
            simulation = Simulation(prmtop.topology, system, integrator, cpu_platform, {})
            simulation.context.setPositions(inpcrd.positions)
            if box_vectors is not None:
                simulation.context.setPeriodicBoxVectors(*box_vectors)
            platform = cpu_platform
            properties = {}
        else:
            raise
    state_before = simulation.context.getState(getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    simulation.minimizeEnergy(maxIterations=int(max_iterations))

    state_after = simulation.context.getState(getEnergy=True, getPositions=True)
    e_after = state_after.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    with open(output_pdb, "w", encoding="utf-8") as f:
        PDBFile.writeFile(prmtop.topology, state_after.getPositions(), f)

    if state_xml_path:
        with open(state_xml_path, "w", encoding="utf-8") as f:
            f.write(XmlSerializer.serialize(state_after))

    summary = {
        "mode": "openmm_amber_minimize",
        "success": True,
        "platform": platform.getName(),
        "available_platforms": available_platforms,
        "prmtop_path": os.path.abspath(prmtop_path),
        "inpcrd_path": os.path.abspath(inpcrd_path),
        "output_pdb": os.path.abspath(output_pdb),
        "state_xml_path": os.path.abspath(state_xml_path) if state_xml_path else None,
        "max_iterations": int(max_iterations),
        "nonbonded_cutoff_nm": float(nonbonded_cutoff_nm),
        "periodic": bool(is_periodic),
        "nonbonded_method": "PME" if is_periodic else "NoCutoff",
        "constraints": str(constraints),
        "temperature_K": float(temperature_K),
        "friction_per_ps": float(friction_per_ps),
        "timestep_fs": float(timestep_fs),
        "energy_before_kj_per_mol": float(e_before),
        "energy_after_kj_per_mol": float(e_after),
        "energy_drop_kj_per_mol": float(e_before - e_after),
    }
    summary["fallback_from"] = fallback_from
    summary["fallback_error"] = fallback_error

    if summary_json_path:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose:
        print("[OPENMM] minimization done")
        print(f"[OPENMM] energy before = {e_before:.6f} kJ/mol")
        print(f"[OPENMM] energy after  = {e_after:.6f} kJ/mol")
        print(f"[OPENMM] saved pdb     = {output_pdb}")
        if state_xml_path:
            print(f"[OPENMM] saved xml     = {state_xml_path}")
        if summary_json_path:
            print(f"[OPENMM] saved summary = {summary_json_path}")

    return {
        "output_pdb": output_pdb,
        "state_xml_path": state_xml_path,
        "summary_json_path": summary_json_path,
        "energy_before_kj_per_mol": float(e_before),
        "energy_after_kj_per_mol": float(e_after),
        "energy_drop_kj_per_mol": float(e_before - e_after),
        "platform": platform.getName(),
        "mode": "openmm_amber_minimize",
    }