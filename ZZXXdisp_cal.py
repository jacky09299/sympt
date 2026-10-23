import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scqubits as scq
scq.settings.NUM_CPUS = 1
import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, h
from lcapy import Circuit


def create_circuit_from_yaml(yaml_string, ext_basis="harmonic"):
    """
    相容性 helper: 若 scqubits 版本較新 (有 from_yaml_string) 則優先使用，
    否則相容舊版 scqubits (Circuit(yaml_string, from_file=False, ...))。
    """
    if hasattr(scq.Circuit, "from_yaml_string"):
        return scq.Circuit.from_yaml_string(yaml_string, ext_basis=ext_basis)
    else:
        return scq.Circuit(yaml_string, from_file=False, ext_basis=ext_basis)


def plot_energy_levels_original(hilbertspace, num_states=15, min_label_dist=0.6, figsize=(8, 10)):
    """
    繪製帶有 bare_index 標記的能階圖。
    使用 ordering='BE' 重新生成 lookup table，確保特徵能量與 bare_index 的物理對應完全精確。
    """
    # 1. 重新生成 lookup table，改用 Bare Energy 排序
    hilbertspace.generate_lookup(ordering='BE', BEs_count=num_states)

    # 2. 取得系統實際的特徵能量 (Dressed Eigenenergies)
    eigenenergies = hilbertspace.eigenvals(evals_count=num_states)
    energy0 = eigenenergies[0]  # 基態能量作為基準 (0 點)

    # 3. 將真實能量與 bare index 配對
    data = []
    for i in range(num_states):
        idx = hilbertspace.bare_index(i)
        energy = eigenenergies[i] - energy0
        data.append({'energy': energy, 'idx': idx})

    # 4. 建立畫布
    fig, ax = plt.subplots(figsize=figsize)

    line_x_range = [0.2, 0.5]
    text_x_pos = 0.6
    last_label_y = -np.inf
    plotted_energies = set()   # 紀錄已標示過的能量數值，避免簡併態字體疊加變粗

    for item in data:
        e_val = item['energy']
        idx = item['idx']

        # 畫出水平能階線
        ax.hlines(e_val, line_x_range[0], line_x_range[1], colors='black', linewidth=1.5)

        # 計算防重疊的標籤位置
        display_y = max(e_val, last_label_y + min_label_dist)

        # 格式化標籤文字 (例如 |0,0,0,0,0>)
        label_str = f"$|{','.join(map(str, idx))}\\rangle$"

        # 畫出右側標籤與指引線
        ax.annotate(
            label_str,
            xy=(line_x_range[1], e_val),
            xytext=(text_x_pos, display_y),
            arrowprops=dict(arrowstyle="->", color='gray', lw=0.8, connectionstyle="arc3,rad=0.1"),
            verticalalignment='center',
            fontsize=11
        )

        # 在最左側標註精確能量數值
        e_str = f"{e_val:.4f}"
        if e_str not in plotted_energies:
            ax.text(line_x_range[0] - 0.02, e_val, e_str,
                    ha='right', va='center', fontsize=9, color='blue', alpha=0.7)
            plotted_energies.add(e_str)

        last_label_y = display_y

    # 5. 圖表美化
    ax.set_title(f"Energy Level Spectrum (Top {num_states} States)", fontsize=14, pad=20)
    ax.set_ylabel("Energy (GHz)", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    return fig, ax


def J12(L, C, L_side, C_side, Cc, EC1, EC2, f1_GHz, f2_GHz):
    values = {
        "L": L,
        "C": C,
        "L_side": L_side,
        "C_side": C_side,
        "Cc": Cc,
        "EC1": EC1,
        "EC2": EC2,
        "f1_GHz": f1_GHz,
        "f2_GHz": f2_GHz,
    }

    invalid = [
        name
        for name, value in values.items()
        if not np.isfinite(value) or value <= 0
    ]
    if invalid:
        raise ValueError(f"下列參數必須是有限且大於 0 的數值：{invalid}")

    EC1_J = h * EC1 * 1e9
    EC2_J = h * EC2 * 1e9

    Cq1 = e**2 / (2 * EC1_J) * 1e15  # fF
    Cq2 = e**2 / (2 * EC2_J) * 1e15  # fF

    netlist = f"""
C1 6 0 {C}f; down
L1 5 6 {L}n; down

Cside3 3 5 {C_side}f; right
Lside3 3 0 {L_side}n; down
Cside4 4 5 {C_side}f; left
Lside4 4 0 {L_side}n; down
Cq1 1 0 {Cq1}f; down
Cc1 1 3 {Cc}f; right
Cq2 2 0 {Cq2}f; down
Cc2 4 2 {Cc}f; right
"""

    cct = Circuit(netlist)
    two_port = cct.twoport("1", "0", "2", "0")

    Z_sym = sm.Matrix(two_port.Zparams)

    s_candidates = [sym for sym in Z_sym.free_symbols if str(sym) == "s"]
    if len(s_candidates) != 1:
        raise ValueError(
            f"無法唯一識別 Lcapy 的 s 變數，找到：{s_candidates}"
        )

    s = s_candidates[0]
    Z12_expr = sm.simplify(Z_sym[0, 1])
    Z12_func = sm.lambdify(s, Z12_expr, modules="numpy")

    w1 = 2 * np.pi * f1_GHz * 1e9
    w2 = 2 * np.pi * f2_GHz * 1e9

    Cq1_SI = Cq1 * 1e-15
    Cq2_SI = Cq2 * 1e-15

    Lq1 = 1 / (Cq1_SI * w1**2)
    Lq2 = 1 / (Cq2_SI * w2**2)

    z12_w1 = complex(np.asarray(Z12_func(1j * w1)).item())
    z12_w2 = complex(np.asarray(Z12_func(1j * w2)).item())

    J12_rad_s = -0.25 * np.sqrt(
        (w1 * w2) / (Lq1 * Lq2)
    ) * np.imag(
        z12_w1 / w1 + z12_w2 / w2
    )

    return J12_rad_s / (2 * np.pi * 1e9)


def disp_ZZ(L, C, L_side=None, C_side=None, Cc=5, EC1=0.2, EJ1=20, EC2=0.2, EJ2=20):
    resonator_description = f"""
    branches:
    - [C,4,0,C_res={C} fF]
    - [L,3,4,L_res={L} nH]
    - [C,1,3,C_res_left={C_side} fF]
    - [L,1,0,L_res_left={L_side} nH]
    - [C,2,3,C_res_right={C_side} fF]
    - [L,2,0,L_res_right={L_side} nH]
    """

    circuit_description = f"""
    branches:
    - [C,6,0,C_res={C} fF]
    - [L,5,6,L_res={L} nH]
    - [C,3,5,C_res_left={C_side} fF]
    - [L,3,0,L_res_left={L_side} nH]
    - [C,4,5,C_res_right={C_side} fF]
    - [L,4,0,L_res_right={L_side} nH]
    - [C,1,0,EC1={EC1} GHz]
    - [JJ,1,0,EJ1={EJ1},0.000000001 fF]
    - [C,1,3,Cc = {Cc} fF]
    - [JJ,2,0,EJ2={EJ2},0.000000001 fF]
    - [C,2,0,EC2={EC2} GHz]
    - [C,4,2,Cc]
    """

    resonator = create_circuit_from_yaml(resonator_description, ext_basis="harmonic")
    T = resonator.transformation_matrix

    I2 = np.eye(2)
    Z24 = np.zeros((2, 4))
    Z42 = np.zeros((4, 2))

    trans_mat = np.block([
        [I2,  Z24],
        [Z42, T]
    ])

    circuit = create_circuit_from_yaml(circuit_description, ext_basis="harmonic")
    circuit.configure(transformation_matrix=trans_mat)

    # 1. 自動辨識變數分類 (periodic: transmon qubits; extended: oscillator modes; free: unconstrained particles)
    periodic_indices = circuit.var_categories["periodic"]
    extended_indices = circuit.var_categories["extended"]
    free_indices = circuit.var_categories["free"]

    print(f"Variable categories -> periodic (qubits): {periodic_indices}, extended (resonators): {extended_indices}, free: {free_indices}")

    coefs = circuit.hamiltonian_symbolic.as_coefficients_dict()

    # 2. SymPy 變數定義
    n_syms = {i: sm.Symbol(f'n{i}') for i in periodic_indices}
    Q_syms = {i: sm.Symbol(f'Q{i}') for i in extended_indices}
    theta_syms = {i: sm.Symbol(f'θ{i}') for i in extended_indices}

    # 3. 建立 Transmon 物件 (針對 periodic 變數)
    EJ_map = {1: EJ1, 2: EJ2}
    qubits = {}
    for i in periodic_indices:
        EC_i = float(coefs.get(n_syms[i]**2, 0)) / 4.0
        EJ_i = EJ_map.get(i, EJ1)
        qubits[i] = scq.Transmon(EJ=EJ_i, EC=EC_i, ng=0.0, ncut=30, truncated_dim=4)

    # 4. 建立 Oscillator 物件 (針對 extended 變數)
    oscillators = {}
    res_E = {}
    res_l_osc = {}
    for idx in extended_indices:
        c_Q2 = float(coefs.get(Q_syms[idx]**2, 0))
        c_theta2 = float(coefs.get(theta_syms[idx]**2, 0))
        E_val = np.sqrt(abs(c_Q2 * c_theta2 * 4))
        l_osc_val = scq.core.oscillator.convert_to_l_osc(c_Q2 * 2, c_theta2 * 2)
        res_E[idx] = E_val
        res_l_osc[idx] = l_osc_val
        oscillators[idx] = scq.Oscillator(E_osc=E_val, l_osc=l_osc_val, truncated_dim=4)

    # 依照諧振子能量由小到大排序
    sorted_ext_indices = sorted(extended_indices, key=lambda x: res_E[x])
    print("Oscillator energies (GHz):", [res_E[idx] for idx in sorted_ext_indices])

    # 5. 組裝 HilbertSpace (順序：qubits 依序 + oscillators 依能量排序)
    subsystems = [qubits[i] for i in periodic_indices] + [oscillators[idx] for idx in sorted_ext_indices]
    hilbertspace = scq.HilbertSpace(subsystems)

    def get_term(op1, op2):
        return float(coefs.get(op1 * op2, 0))

    # Transmon 之間的耦合 (n * n)
    for idx_a, i in enumerate(periodic_indices):
        for j in periodic_indices[idx_a + 1:]:
            g_val = get_term(n_syms[i], n_syms[j])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(g=g_val, op1=qubits[i].n_operator, op2=qubits[j].n_operator)

    # Transmon (n) 與 Resonator (Q) 之間的耦合
    for i in periodic_indices:
        for idx in sorted_ext_indices:
            g_val = get_term(n_syms[i], Q_syms[idx])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(g=g_val, op1=qubits[i].n_operator, op2=oscillators[idx].n_operator)

    # Resonator (Q) 之間的耦合 (Q * Q)
    for idx_a, idx1 in enumerate(sorted_ext_indices):
        for idx2 in sorted_ext_indices[idx_a + 1:]:
            g_val = get_term(Q_syms[idx1], Q_syms[idx2])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(g=g_val, op1=oscillators[idx1].n_operator, op2=oscillators[idx2].n_operator)

    num_subsystems = len(subsystems)
    subsys_priority = list(reversed(range(num_subsystems)))
    hilbertspace.generate_lookup(ordering='LX', subsys_priority=subsys_priority)

    # 6. 計算能量與 ZZ, XX, Dispersive Shift
    # 態索引格式: (q1, q2, r1, r2, r3)
    E_00000 = hilbertspace.energy_by_bare_index((0, 0, 0, 0, 0))
    E_10000 = hilbertspace.energy_by_bare_index((1, 0, 0, 0, 0))
    E_01000 = hilbertspace.energy_by_bare_index((0, 1, 0, 0, 0))
    E_11000 = hilbertspace.energy_by_bare_index((1, 1, 0, 0, 0))

    ZZ = E_11000 - E_01000 - E_10000 + E_00000

    E_00100 = hilbertspace.energy_by_bare_index((0, 0, 1, 0, 0))
    E_00010 = hilbertspace.energy_by_bare_index((0, 0, 0, 1, 0))

    E_10100 = hilbertspace.energy_by_bare_index((1, 0, 1, 0, 0))
    E_01100 = hilbertspace.energy_by_bare_index((0, 1, 1, 0, 0))

    E_10010 = hilbertspace.energy_by_bare_index((1, 0, 0, 1, 0))
    E_01010 = hilbertspace.energy_by_bare_index((0, 1, 0, 1, 0))

    fq1 = E_10000 - E_00000
    fq2 = E_01000 - E_00000
    fc1 = E_00100 - E_00000
    fc2 = E_00010 - E_00000

    XX = J12(L, C, L_side, C_side, Cc, EC1, EC2, fq1, fq2)
    print("XX: ", (E_10000 - E_01000) / 2)

    disp_10_3 = (E_00100 - E_00000) - (E_10100 - E_10000)
    disp_01_3 = (E_00100 - E_00000) - (E_01100 - E_01000)

    disp_10_4 = (E_00010 - E_00000) - (E_10010 - E_10000)
    disp_01_4 = (E_00010 - E_00000) - (E_01010 - E_01000)

    return fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4


if __name__ == "__main__":
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2 = 2, 114, 47.98418086298829, 5.0, 6.495487626697703, 0.1426, 20,0.1426, 20

    fq1, fq2, fc1, fc2, ZZ, XX, disp_10_3, disp_01_3, disp_10_4, disp_01_4 = disp_ZZ(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2)
    print(f"ZZ: {ZZ*1000:.6g}")
    print(f"XX: {XX*1000:.6g}")
    print(f"fq1: {fq1:.3g}")
    print(f"fq2: {fq2:.3g}")
    print(f"fc1: {fc1:.3g}")
    print(f"fc2: {fc2:.3g}")
    print(f"disp_10_3: {disp_10_3*1000:.4g}")
    print(f"disp_01_3: {disp_01_3*1000:.4g}")
    print(f"disp_10_4: {disp_10_4*1000:.4g}")
    print(f"disp_01_4: {disp_01_4*1000:.4g}")
