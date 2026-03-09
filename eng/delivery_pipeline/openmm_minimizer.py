import os
import math
import json

from openmm.app import PDBFile, Simulation
from openmm import System, CustomNonbondedForce, VerletIntegrator, Platform, Vec3
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
def compute_packaging_metrics(
    pdb_path: str,
    core_radius_A: float = 125.0,
    drug_chain_id: str = "D",
    atoms_per_drug: int = 57,
):
    """
    计算 4 个简单但有用的包装指标：
    1. 装入率 f_in
    2. 最小边界余量 delta_min_A
    3. 药物质心半径分布 drug_radius_A
    4. 药物最近邻距离 drug_nn_dist_A

    约定（按你当前体系）：
    - 药物链是 D
    - 每个药物 57 个原子
    - 核心半径 125 Å
    """

    pdb = PDBFile(pdb_path)
    atoms = list(pdb.topology.atoms())

    # 全体系坐标（Å）
    all_positions_A = [_vec3_to_A(p) for p in pdb.positions]
    nanoparticle_center = _mean_xyz(all_positions_A)

    # 取药物链原子
    drug_positions_A = []
    for atom, pos in zip(atoms, all_positions_A):
        if atom.residue.chain.id == drug_chain_id:
            drug_positions_A.append(pos)

    if not drug_positions_A:
        raise RuntimeError(
            f"在 PDB 中没有找到药物链 '{drug_chain_id}'，无法计算包装指标。"
        )

    if len(drug_positions_A) % atoms_per_drug != 0:
        raise RuntimeError(
            f"药物链 '{drug_chain_id}' 原子数 = {len(drug_positions_A)}，"
            f"不能被 atoms_per_drug = {atoms_per_drug} 整除。"
        )

    n_drugs = len(drug_positions_A) // atoms_per_drug

    drug_centers = []
    boundary_margins = []
    inside_count = 0

    # 逐个药物切块
    for i in range(n_drugs):
        block = drug_positions_A[i * atoms_per_drug:(i + 1) * atoms_per_drug]

        # 药物质心
        c = _mean_xyz(block)
        drug_centers.append(c)

        # 该药物最外层原子到纳米颗粒中心的最大半径
        r_max = max(_norm3(p, nanoparticle_center) for p in block)

        # 边界余量 Δ_i = R_core - r_max
        delta_i = float(core_radius_A - r_max)
        boundary_margins.append(delta_i)

        if delta_i > 0.0:
            inside_count += 1

    # 1) 装入率
    f_in = float(inside_count / n_drugs)

    # 2) 最小边界余量
    delta_min_A = float(min(boundary_margins))

    # 3) 药物质心半径分布
    drug_radius_A = [float(_norm3(c, nanoparticle_center)) for c in drug_centers]

    # 4) 药物最近邻距离
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
        "delta_min_A": delta_min_A,
        "drug_radius_mean_A": _mean_value(drug_radius_A),
        "drug_nn_dist_mean_A": _mean_value(drug_nn_dist_A),
    }


def run_openmm_minimization(
    input_pdb: str,
    output_pdb: str,
    max_iterations: int = 2000,
    repulsion_k: float = 10.0,     # kJ/mol/nm^2
    sigma_nm: float = 0.25,        # nm
    cutoff_nm: float = 1.0,        # nm
    platform_name: str = "CPU",
    stages: int = 3,
    verbose: bool = False,
):
    """
    用 OpenMM 做“软排斥”规整（处理 PACKMOL 的近距离碰撞）：
    - 不依赖力场（不需要 forcefield 文件）
    - 只做软排斥：step(sigma-r) * k * (sigma-r)^2
    - 分阶段增大 k / 缩小 sigma，让体系更稳更不容易炸

    单位约定（OpenMM 的默认 nm / kJ/mol 体系）：
    - repulsion_k：k（kJ/mol/nm^2）
    - sigma_nm：sigma（nm）
    - cutoff_nm：cutoff（nm）
    """

    _ensure_dir(output_pdb)

    pdb = PDBFile(input_pdb)
    topology = pdb.topology
    positions = pdb.positions

    atoms = list(topology.atoms())
    n_atoms = len(atoms)

    if verbose:
        print(f"[OPENMM] start minimization: N_atoms={n_atoms}, max_iter={max_iterations}, cutoff_nm={cutoff_nm}")
    # -----------------------------
    # system: 只加粒子质量即可（不跑动力学，只做最小化）
    # -----------------------------
    system = System()
    for _ in atoms:
        system.addParticle(12.0)  # dalton(amu)，任意正数都可

    # -----------------------------
    # Soft repulsion
    # -----------------------------
    energy = "step(sigma - r) * k * (sigma - r)^2"
    force = CustomNonbondedForce(energy)
    force.addGlobalParameter("k", float(repulsion_k))
    force.addGlobalParameter("sigma", float(sigma_nm))

    force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(float(cutoff_nm) * unit.nanometer)

    for _ in range(n_atoms):
        force.addParticle([])

    system.addForce(force)

    integrator = VerletIntegrator(0.001 * unit.picoseconds)

    try:
        platform = Platform.getPlatformByName(platform_name)
    except Exception as e:
        names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        raise RuntimeError(
            f"Platform '{platform_name}' not available. Available platforms: {names}. Original error: {e}"
        )

    properties = {}
    if platform.getName() in ("CUDA", "OpenCL"):
        properties["Precision"] = "mixed"

    simulation = Simulation(topology, system, integrator, platform, properties)

    import random

    if hasattr(positions, "value_in_unit"):
        positions_nm = positions.value_in_unit(unit.nanometer)
    else:
        positions_nm = positions

    jitter_nm = 0.01
    pos_jitter_nm = []
    for p in positions_nm:
        dx = random.uniform(-1.0, 1.0) * jitter_nm
        dy = random.uniform(-1.0, 1.0) * jitter_nm
        dz = random.uniform(-1.0, 1.0) * jitter_nm
        pos_jitter_nm.append(Vec3(p.x + dx, p.y + dy, p.z + dz))

    pos_jitter = [v * unit.nanometer for v in pos_jitter_nm]
    simulation.context.setPositions(pos_jitter)

    if stages < 1:
        stages = 1

    sigma_start = sigma_nm * 1.8
    k_start = repulsion_k * 0.2

    for s in range(stages):
        if stages == 1:
            sigma_s = sigma_nm
            k_s = repulsion_k
            it_s = max_iterations
        else:
            t = s / (stages - 1)
            sigma_s = sigma_start * (1.0 - t) + sigma_nm * t
            k_s = k_start * (1.0 - t) + repulsion_k * t
            it_s = max(200, int(max_iterations / stages))

        simulation.context.setParameter("sigma", float(sigma_s))
        simulation.context.setParameter("k", float(k_s))

        if verbose:
            print(f"[OPENMM] stage {s + 1}/{stages}: sigma_nm={sigma_s:.4f}, k={k_s:.4f}, it={it_s}")
        simulation.minimizeEnergy(
            tolerance=10.0 * unit.kilojoule / (unit.mole * unit.nanometer),
            maxIterations=int(it_s)
        )

    state = simulation.context.getState(getPositions=True)

    with open(output_pdb, "w", encoding="utf-8") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)

    print("[OPENMM] minimization done")
    print("[OPENMM] saved:", output_pdb)
    return output_pdb