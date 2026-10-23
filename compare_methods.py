"""
compare_methods.py

比對三種方法在兩顆完全相同 Transmon (Identical Qubits) 條件下計算 XX Coupling (J_XX) 的數值結果：
1. 能階法 (Energy Level Method, scqubits 確切對角化): J_XX = (E_10000 - E_01000) / 2
2. 黑箱阻抗法 (Black-Box Impedance Method, J12 lcapy 雙埠 Z 參數)
3. SymPT 微擾法 (Schrieffer-Wolff 轉換: 2nd Order RWA & Full)
"""

import numpy as np
import scqubits as scq
import sympy as sm

from ZZXXdisp_cal import J12, create_circuit_from_yaml
from ZZXXdisp_cal_sympt import calc_XX_sympt


def get_energy_level_Jxx(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2):
    """
    使用能階法 (HilbertSpace 精確對角化) 計算 J_XX = (E_10000 - E_01000) / 2
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

    resonator_description = f"""
    branches:
    - [C,4,0,C_res={C} fF]
    - [L,3,4,L_res={L} nH]
    - [C,1,3,C_res_left={C_side} fF]
    - [L,1,0,L_res_left={L_side} nH]
    - [C,2,3,C_res_right={C_side} fF]
    - [L,2,0,L_res_right={L_side} nH]
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

    periodic_indices = circuit.var_categories["periodic"]
    extended_indices = circuit.var_categories["extended"]

    coefs = circuit.hamiltonian_symbolic.as_coefficients_dict()

    n_syms = {i: sm.Symbol(f'n{i}') for i in periodic_indices}
    Q_syms = {i: sm.Symbol(f'Q{i}') for i in extended_indices}
    theta_syms = {i: sm.Symbol(f'θ{i}') for i in extended_indices}

    EJ_map = {1: EJ1, 2: EJ2}
    qubits = {}
    for i in periodic_indices:
        EC_i = float(coefs.get(n_syms[i]**2, 0)) / 4.0
        EJ_i = EJ_map.get(i, EJ1)
        qubits[i] = scq.Transmon(EJ=EJ_i, EC=EC_i, ng=0.0, ncut=30, truncated_dim=4)

    oscillators = {}
    res_E = {}
    for idx in extended_indices:
        c_Q2 = float(coefs.get(Q_syms[idx]**2, 0))
        c_theta2 = float(coefs.get(theta_syms[idx]**2, 0))
        E_val = np.sqrt(abs(c_Q2 * c_theta2 * 4))
        l_osc_val = scq.core.oscillator.convert_to_l_osc(c_Q2 * 2, c_theta2 * 2)
        res_E[idx] = E_val
        oscillators[idx] = scq.Oscillator(E_osc=E_val, l_osc=l_osc_val, truncated_dim=4)

    sorted_ext_indices = sorted(extended_indices, key=lambda x: res_E[x])
    subsystems = [qubits[i] for i in periodic_indices] + [oscillators[idx] for idx in sorted_ext_indices]
    hilbertspace = scq.HilbertSpace(subsystems)

    def get_term(op1, op2):
        return float(coefs.get(op1 * op2, 0))

    for idx_a, i in enumerate(periodic_indices):
        for j in periodic_indices[idx_a + 1:]:
            g_val = get_term(n_syms[i], n_syms[j])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(g=g_val, op1=qubits[i].n_operator, op2=qubits[j].n_operator)

    for i in periodic_indices:
        for idx in sorted_ext_indices:
            g_val = get_term(n_syms[i], Q_syms[idx])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(g=g_val, op1=qubits[i].n_operator, op2=oscillators[idx].n_operator)

    for idx_a, idx1 in enumerate(sorted_ext_indices):
        for idx2 in sorted_ext_indices[idx_a + 1:]:
            g_val = get_term(Q_syms[idx1], Q_syms[idx2])
            if abs(g_val) > 1e-12:
                hilbertspace.add_interaction(g=g_val, op1=oscillators[idx1].n_operator, op2=oscillators[idx2].n_operator)

    num_subsystems = len(subsystems)
    subsys_priority = list(reversed(range(num_subsystems)))
    hilbertspace.generate_lookup(ordering='LX', subsys_priority=subsys_priority)

    E_00000 = hilbertspace.energy_by_bare_index((0, 0, 0, 0, 0))
    E_10000 = hilbertspace.energy_by_bare_index((1, 0, 0, 0, 0))
    E_01000 = hilbertspace.energy_by_bare_index((0, 1, 0, 0, 0))

    fq1 = E_10000 - E_00000
    fq2 = E_01000 - E_00000

    Jxx_el = abs(E_10000 - E_01000) / 2.0
    return Jxx_el, fq1, fq2


def run_comparison():
    # 電路參數設定 (兩顆 Transmon 一模一樣: EC1=EC2=0.1426, EJ1=EJ2=20)
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2 = 2, 114, 47.98418086298829, 5.0, 6.495487626697703, 0.1426, 20, 0.1426, 20

    print("=========================================================================")
    print(" XX Coupling (J_XX) 三大計算方法數值對比分析 (Identical Qubits)")
    print("=========================================================================")
    print(f" Transmon 參數: EC1 = EC2 = {EC1} GHz, EJ1 = EJ2 = {EJ1} GHz")
    print("-------------------------------------------------------------------------")

    # 1. 能階法 (Ground Truth)
    print("1. 正在執行能階法 (scqubits HilbertSpace 精確對角化)...")
    jxx_el_GHz, fq1, fq2 = get_energy_level_Jxx(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2)
    jxx_el_MHz = jxx_el_GHz * 1000.0

    # 2. 黑箱阻抗法 (J12)
    print("2. 正在執行黑箱阻抗法 (Lcapy Z 參數 J12)...")
    jxx_bb_GHz = J12(L, C, L_side, C_side, Cc, EC1, EJ1, fq1, fq2)
    jxx_bb_MHz = jxx_bb_GHz * 1000.0

    # 3. SymPT 法 (Order 2 RWA)
    print("3. 正在執行 SymPT 法 (2nd Order SW - RWA)...")
    jxx_sympt_rwa_GHz = calc_XX_sympt(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=2, mode="rwa")
    jxx_sympt_rwa_MHz = jxx_sympt_rwa_GHz * 1000.0

    # 4. SymPT 法 (Order 2 Full)
    print("4. 正在執行 SymPT 法 (2nd Order SW - Full)...")
    jxx_sympt_full_GHz = calc_XX_sympt(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=2, mode="full")
    jxx_sympt_full_MHz = jxx_sympt_full_GHz * 1000.0

    print("\n=========================================================================")
    print("                            數值對比總表                                 ")
    print("=========================================================================")
    headers = f"{'方法 (Method)':<32} | {'J_XX (MHz)':<14} | {'絕對誤差 (MHz)':<15} | {'相對誤差 (%)':<15}"
    print(headers)
    print("-" * len(headers))

    methods = [
        ("能階法 [(E10000 - E01000) / 2]", jxx_el_MHz),
        ("黑箱阻抗法 (Black-Box J12)", jxx_bb_MHz),
        ("SymPT (2nd Order SW - RWA)", jxx_sympt_rwa_MHz),
        ("SymPT (2nd Order SW - Full)", jxx_sympt_full_MHz),
    ]

    for name, val_mhz in methods:
        abs_err = abs(abs(val_mhz) - jxx_el_MHz)
        rel_err = (abs_err / jxx_el_MHz) * 100.0
        print(f"{name:<32} | {val_mhz:<14.6f} | {abs_err:<15.6f} | {rel_err:<15.2f}%")

    print("=========================================================================")
    print(" 分析說明:")
    print(f" - 能階法 (Ground Truth): J_XX = {jxx_el_MHz:.6f} MHz。")
    print(f" - 黑箱阻抗法 J12: J_XX = {jxx_bb_MHz:.6f} MHz。")
    print(f" - SymPT 二階 SW 轉換 (RWA 模式): J_XX = {jxx_sympt_rwa_MHz:.6f} MHz。")
    print(f" - SymPT 二階 SW 轉換 (Full 模式): J_XX = {jxx_sympt_full_MHz:.6f} MHz。")
    print("=========================================================================")


if __name__ == "__main__":
    run_comparison()
