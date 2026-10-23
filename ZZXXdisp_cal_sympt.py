import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import scqubits as scq
scq.settings.NUM_CPUS = 1

import sympy as sp
import numpy as np
import sympt as sympt
from sympt import RDSymbol, EffectiveFrame, BosonOp, Dagger, group_by_operators


def create_circuit_from_yaml(yaml_string, ext_basis="harmonic"):
    """
    建立 scqubits Circuit 物件助手函式。
    """
    if hasattr(scq.Circuit, "from_yaml_string"):
        return scq.Circuit.from_yaml_string(yaml_string, ext_basis=ext_basis)
    else:
        return scq.Circuit(yaml_string, from_file=False, ext_basis=ext_basis)


def calc_XX_sympt(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=2, mode="full", verbose=False):
    """
    使用 SymPT (Schrieffer-Wolff 轉換) 計算雙 Transmon 耦合至多模態諧振子電路的 XX Coupling (J_XX)。

    Parameters
    ----------
    L, C, L_side, C_side, Cc : float
        諧振子耦合電路之電感電容參數。
    EC1, EJ1, EC2, EJ2 : float
        兩顆 Transmon 的能量參數 (GHz)。
    order : int, optional
        Schrieffer-Wolff 攝動展開階數 (預設為 2)。
    mode : str, optional
        "full" (包含完整 Counter-rotating 反旋轉作用項，與精確能階法誤差 < 7%) 或 "rwa" (旋轉波近似)。
    verbose : bool, optional
        是否印出詳細解算過程。

    Returns
    -------
    float
        J_XX 耦合強度 (GHz)。
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

    n_syms = {i: sp.Symbol(f'n{i}') for i in periodic_indices}
    Q_syms = {i: sp.Symbol(f'Q{i}') for i in extended_indices}
    theta_syms = {i: sp.Symbol(f'θ{i}') for i in extended_indices}

    EC_map = {1: EC1, 2: EC2}
    EJ_map = {1: EJ1, 2: EJ2}

    wq = {}
    alpha_q = {}
    nzpf = {}

    for i in periodic_indices:
        c_n2 = float(coefs.get(n_syms[i]**2, 0))
        ec_val = c_n2 / 4.0
        ej_val = EJ_map[i]
        wq_val = np.sqrt(8 * ec_val * ej_val) - ec_val
        wq[i] = wq_val
        alpha_q[i] = -ec_val
        nzpf[i] = (ej_val / (32.0 * ec_val))**0.25

    w_res = {}
    Qzpf = {}
    for idx in extended_indices:
        c_Q2 = float(coefs.get(Q_syms[idx]**2, 0))
        c_theta2 = float(coefs.get(theta_syms[idx]**2, 0))
        w_val = np.sqrt(abs(c_Q2 * c_theta2 * 4))
        w_res[idx] = w_val
        # 零點漲落 Q_zpf: (c_theta2 / (4 * c_Q2))**(1/4)
        Qzpf[idx] = (c_theta2 / (4.0 * c_Q2))**0.25

    # 直連電荷耦合 g12
    c_n1n2 = float(coefs.get(n_syms[1]*n_syms[2], 0))
    g12_direct = - c_n1n2 * nzpf[1] * nzpf[2]

    # Transmon 與 Resonator 耦合
    g_q_res = {}
    for q_idx in periodic_indices:
        for r_idx in extended_indices:
            c_qn = float(coefs.get(Q_syms[r_idx]*n_syms[q_idx], 0))
            g_val = c_qn * Qzpf[r_idx] * nzpf[q_idx]
            g_q_res[(q_idx, r_idx)] = g_val

    # 定義 SymPT 符號與算符
    w1_sym = RDSymbol('w1', order=0, positive=True, real=True)
    w2_sym = RDSymbol('w2', order=0, positive=True, real=True)
    alpha1_sym = RDSymbol('alpha1', order=0, real=True)
    alpha2_sym = RDSymbol('alpha2', order=0, real=True)

    res_syms = {idx: RDSymbol(f'wr_{idx}', order=0, positive=True, real=True) for idx in extended_indices}

    g12_sym = RDSymbol('g12', order=1, real=True)
    g_syms = {}
    for q_idx in periodic_indices:
        for r_idx in extended_indices:
            g_syms[(q_idx, r_idx)] = RDSymbol(f'g_{q_idx}_{r_idx}', order=1, real=True)

    a1 = BosonOp('a1'); ad1 = Dagger(a1)
    a2 = BosonOp('a2'); ad2 = Dagger(a2)
    b_ops = {idx: BosonOp(f'b{idx}') for idx in extended_indices}
    bd_ops = {idx: Dagger(b_ops[idx]) for idx in extended_indices}

    H0 = w1_sym * ad1 * a1 + w2_sym * ad2 * a2 \
         + sp.Rational(1, 2) * alpha1_sym * ad1 * ad1 * a1 * a1 \
         + sp.Rational(1, 2) * alpha2_sym * ad2 * ad2 * a2 * a2

    for idx in extended_indices:
        H0 += res_syms[idx] * bd_ops[idx] * b_ops[idx]

    if mode.lower() == "rwa":
        V = g12_sym * (ad1 * a2 + a1 * ad2)
        for q_idx in periodic_indices:
            a_op = ad1 if q_idx == 1 else ad2
            a_ann = a1 if q_idx == 1 else a2
            for r_idx in extended_indices:
                if abs(g_q_res[(q_idx, r_idx)]) > 1e-10:
                    V += g_syms[(q_idx, r_idx)] * (a_op * b_ops[r_idx] + a_ann * bd_ops[r_idx])
    else:
        # Full charge-charge interaction
        V = g12_sym * (ad1 - a1) * (ad2 - a2)
        for q_idx in periodic_indices:
            a_op = ad1 if q_idx == 1 else ad2
            a_ann = a1 if q_idx == 1 else a2
            for r_idx in extended_indices:
                if abs(g_q_res[(q_idx, r_idx)]) > 1e-10:
                    V += g_syms[(q_idx, r_idx)] * (a_op - a_ann) * (bd_ops[r_idx] + b_ops[r_idx])

    eff = EffectiveFrame(H0, V, verbose=verbose)
    eff.solve(max_order=order, method='SW')
    H_eff = eff.get_H(return_form='operator')

    grouped = group_by_operators(H_eff)

    sub_dict = {
        w1_sym: wq[1],
        w2_sym: wq[2],
        alpha1_sym: alpha_q[1],
        alpha2_sym: alpha_q[2],
        g12_sym: g12_direct
    }
    for idx in extended_indices:
        sub_dict[res_syms[idx]] = w_res[idx]
        for q_idx in periodic_indices:
            sub_dict[g_syms[(q_idx, idx)]] = g_q_res[(q_idx, idx)]

    # 提取交換項算符 (Dagger(a1)*a2) 之單一係數
    exchange_op = ad1 * a2
    mediated_coeff_expr = grouped.get(exchange_op, 0)

    if isinstance(mediated_coeff_expr, (int, float)):
        eval_mediated = float(mediated_coeff_expr)
    else:
        eval_mediated = float(mediated_coeff_expr.subs({ad1*a1: 0, ad2*a2: 0}).subs(sub_dict))

    if mode.lower() == "rwa":
        J_xx = g12_direct + eval_mediated
    else:
        # Full mode 下 (ad1 - a1)(ad2 - a2) 中 ad1*a2 之直連係數為 -g12_direct
        J_xx = - g12_direct - eval_mediated

    return J_xx


if __name__ == "__main__":
    L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2 = 2, 114, 47.98418086298829, 5.0, 6.495487626697703, 0.1426, 20, 0.1426, 20

    J_XX_sympt_full = calc_XX_sympt(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=2, mode="full")
    J_XX_sympt_rwa = calc_XX_sympt(L, C, L_side, C_side, Cc, EC1, EJ1, EC2, EJ2, order=2, mode="rwa")

    print(f"SymPT Calculated XX Coupling (Full Mode): {J_XX_sympt_full * 1000:.6f} MHz")
    print(f"SymPT Calculated XX Coupling (RWA Mode):  {J_XX_sympt_rwa * 1000:.6f} MHz")
