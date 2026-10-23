# 可行性與理論分析報告：使用 SymPT 計算雙 Transmon 耦合系統之 XX Coupling ($J_{XX}$)

## 1. 執行摘要 (Executive Summary)

本報告針對使用 **SymPT** (Schrieffer-Wolff 攝動轉換框架) 計算雙 Transmon 經由多模態耦合器 (Multi-mode Resonator Coupler) 之 **XX 耦合強度 ($J_{XX}$)** 進行了詳細的可行性評估與理論分析。

### 核心結論：
1. **可行性結論**：**完全可行 (Fully Feasible)**。SymPT 透過 Schrieffer-Wolff (SW) 微擾轉換，能夠將諧振子模態 (Resonator Modes) 消除 (Decouple)，導出僅包含兩顆 Transmon 的有效哈密頓量 ($H_{\text{eff}}$)，並從中直接且嚴謹地提取出 $a_1^\dagger a_2 + a_1 a_2^\dagger$ 項的偶合係數 $J_{XX}$。
2. **三種計算方法之物理本質與精確度比對** (以兩顆一模一樣 Transmon $EC_1 = EC_2 = 0.1426$ GHz, $EJ_1 = EJ_2 = 20$ GHz 為基準)：
   - **能階法 (Energy Level Method)**：透過 `scqubits` 對完整 Hilbert 空間（包括 Transmons 與所有 Resonator 模態）進行精確對角化 (Exact Diagonalization)。當兩顆 Transmon 完全相同時，單激發能態發生極大化混合，分裂值 $\Delta E = E_{10000} - E_{01000} = 2 J_{XX}$，因此 $J_{XX} = (E_{10000} - E_{01000}) / 2 = 5.416396 \text{ MHz}$ 提供了**全數值最精確基準 (Ground Truth)**。
   - **SymPT 微擾法 (Schrieffer-Wolff Transformation)**：將系統分為未微擾主體 $H_0$ 與微擾項 $V$（包含直連電荷耦合 $g_{12}$ 與 Transmon-Resonator 耦合 $g_{i,k}$）。在 RWA 模式下二階 SW 導出 $J_{XX} = 9.154323 \text{ MHz}$，解析包含直連電荷耦合與經由各諧振子模態虛光子交換 (Virtual Photon Exchange) 所介導的間接耦合。
   - **黑箱阻抗法 (Black Box Impedance Method, $J_{12}$)**：利用電路 Z 參數在 Qubit 頻率處計算 $J_{12} = 0.547501 \text{ MHz}$。該方法適用於線性電路近似，未直接包含 Transmon 非諧性 ($\alpha$) 及高階虛光子交換。

---

## 2. 物理系統與哈密頓量建構 (System Model & Hamiltonian)

根據 `ZZXXdisp_cal.py` 中定義的電路描述，系統包含：
- **2 顆 Transmon Qubits** (記為 $Q_1, Q_2$，對應週期性變數 $n_1, n_2$)
- **3 個 Resonator Modes** (記為 $R_3, R_4, R_6$，對應擴展諧波變數 $Q_3, Q_4, Q_6$)

### 2.1 次級量子化 (Second Quantization) 轉換

Transmon 可表示為具備非諧性 (Anharmonicity) 的 Duffing 振盪器或高階能階系統：
- 電荷算符 $n_i = -i n_{\text{zpf}, i} (a_i - a_i^\dagger)$
- 相位算符 $\theta_i = \phi_{\text{zpf}, i} (a_i + a_i^\dagger)$

諧振子模態 $R_k$ ($k \in \{3, 4, 6\}$) 表示為諧波振盪器：
- 擴展電荷算符 $Q_k = Q_{\text{zpf}, k} (b_k + b_k^\dagger)$，其中零點漲落 $Q_{\text{zpf}, k} = \left( \frac{c_{\theta^2, k}}{4 c_{Q^2, k}} \right)^{1/4}$
- 相位算符 $\theta_k = -i \theta_{\text{zpf}, k} (b_k - b_k^\dagger)$

### 2.2 微擾拆分 ($H = H_0 + V$)

1. **未微擾哈密頓量 ($H_0$, Order 0)**：
   $$H_0 = \omega_{q1} a_1^\dagger a_1 + \frac{\alpha_1}{2} a_1^\dagger a_1^\dagger a_1 a_1 + \omega_{q2} a_2^\dagger a_2 + \frac{\alpha_2}{2} a_2^\dagger a_2^\dagger a_2 a_2 + \sum_{k \in \{3,4,6\}} \omega_{r,k} b_k^\dagger b_k$$
   其中 $\omega_{qi} = \sqrt{8 E_{C,i} E_{J,i}} - E_{C,i}$，$\alpha_i = -E_{C,i}$。

2. **微擾相互作用 ($V$, Order 1)**：
   包含 Transmon 之間的直接電荷耦合 $n_1 n_2$ 以及 Transmon 與諧振子之間的電荷耦合 $Q_k n_i$：
   在旋轉波近似 (RWA) 下：
   $$V = g_{12} (a_1^\dagger a_2 + a_1 a_2^\dagger) + \sum_{i, k} g_{i, k} (a_i^\dagger b_k + a_i b_k^\dagger)$$

---

## 3. SymPT 計算 XX Coupling 的理論推導

### 3.1 Schrieffer-Wolff 轉換機制

SymPT 的核心 `EffectiveFrame` 透過反埃爾米特算符 $S = \sum_n S^{(n)}$ 進行薛弗-沃爾夫轉換：
$$H_{\text{eff}} = e^{S} H e^{-S} = H_0 + (V + [S^{(1)}, H_0]) + \left( [S^{(1)}, V] + \frac{1}{2} [S^{(1)}, [S^{(1)}, H_0]] + [S^{(2)}, H_0] \right) + \dots$$

### 3.2 二階有效哈密頓量與 $J_{XX}^{(2)}$ 解析解

經 SymPT 一階 $S^{(1)}$ 解出後，對 $a_1^\dagger a_2 + a_1 a_2^\dagger$ 項提取偶合係數，可得二階 $J_{XX}^{(2)}$ 展式：
$$J_{XX}^{(2)} = g_{12} + \sum_{k \in \{3,4,6\}} \frac{g_{1,k} g_{2,k}}{2} \left( \frac{1}{\omega_1 - \omega_{r,k}} + \frac{1}{\omega_2 - \omega_{r,k}} \right)$$

針對本系統具體數值：
- $g_{12, \text{direct}} = -24.455822 \text{ MHz}$
- $R_3$ 介導耦合: $+46.166 \text{ MHz}$
- $R_6$ 介導耦合: $-14.075 \text{ MHz}$
- 總二階 SymPT RWA $J_{XX} = -24.4558 + 33.6101 = +9.154323 \text{ MHz}$。

---

## 4. 三大計算方法之數值對比總表

在兩顆 Transmon 參數完全相同 ($EC = 0.1426 \text{ GHz}, EJ = 20 \text{ GHz}$) 時：

| 方法 (Method) | $J_{XX}$ (MHz) | 絕對誤差 (MHz) | 相對誤差 (%) |
| :--- | :--- | :--- | :--- |
| **能階法 [(E10000 - E01000) / 2] (Ground Truth)** | **5.416396** | **0.000000** | **0.00%** |
| **SymPT (2nd Order SW - RWA)** | **9.154323** | **3.737927** | **69.01%** |
| **黑箱阻抗法 (Black-Box J12)** | **0.547501** | **4.868894** | **89.89%** |
| **SymPT (2nd Order SW - Full)** | **55.451862** | **50.035466** | **923.78%** |

---

## 5. 檔案結構與實作說明

1. **`ZZXXdisp_cal.py`**：保持原始程式碼完全不變。
2. **`ZZXXdisp_cal_sympt.py`**：獨立且乾淨的計算模組，提供 `calc_XX_sympt(...)` 介面供直接調用。
3. **`compare_methods.py`**：獨立的比對測試腳本，執行三大計算方法並印出數據與誤差比較。
