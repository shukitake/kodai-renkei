import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum
import pandas as pd


# ゴキブリの分布をエクセルファイルから読み込む関数
def load_cockroach_distribution(file) -> np.ndarray:
    """
    エクセルファイルからゴキブリの分布を読み込む関数

    Parameters:
        file: アップロードされたエクセルファイル

    Returns:
        np.ndarray: ゴキブリの数を含む行列
    """
    df = pd.read_excel(file, header=None)
    return df.values

# 入力データ作成、ゴキブリ・殺虫剤設置可能点生成
def make_data(n: int, upper: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    ランダムなゴキブリ・殺虫剤の設置可能点を生成する関数

    Parameters:
        n (int): 生成する点の数
        upper (int): 点の座標の上限
        seed (int, optional): 乱数生成の種。デフォルトは0

    Returns:
        Tuple[np.ndarray, np.ndarray]: x座標とy座標の配列
    """
    np.random.seed(seed)  # 乱数シードを設定
    X = np.random.randint(0, upper, size=n)
    Y = np.random.randint(0, upper, size=n)
    return X, Y

# ゴキブリ(p1)と殺虫剤(p2)の距離を測る
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    2点間のユークリッド距離を計算する関数

    Parameters:
        p1 (np.ndarray): ゴキブリの座標
        p2 (np.ndarray): 殺虫剤の座標

    Returns:
        float: 2点間の距離
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

# 需要変数(二値変数)行列を生成
def make_matrix(ps: Tuple[np.ndarray, np.ndarray], r: float) -> np.ndarray:
    """
    ゴキブリと殺虫剤の距離に基づいて、需要変数行列を生成する関数

    Parameters:
        ps (Tuple[np.ndarray, np.ndarray]): ゴキブリと殺虫剤の座標
        r (float): 殺虫剤の効果範囲

    Returns:
        np.ndarray: 需要変数行列
    """
    def check(a: float) -> int:
        """
        距離が効果範囲内かどうかを判定する関数

        Parameters:
            a (float): 距離

        Returns:
            int: 1または0
        """
        return 1 if a <= r else 0

    unit_list = [np.array([n, m]) for n, m in zip(ps[0], ps[1])]
    return np.array([[check(distance(p1, p2)) for p1 in unit_list] for p2 in unit_list])

# パラメータを生成する関数
@st.cache_data
def generate_parameters(
    a: int, b: int, seed: int = 0, radius: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ゴキブリの数と需要変数の行列を生成する関数

    Parameters:
        a (int): グリッドの行数
        b (int): グリッドの列数
        seed (int, optional): 乱数生成の種。デフォルトは0
        radius (float, optional): 殺虫剤の効果範囲。デフォルトは5.0

    Returns:
        Tuple[np.ndarray, np.ndarray]: ゴキブリの数を含む行列Wと需要変数行列A
    """
    np.random.seed(seed)
    W = np.random.randint(0, 3, size=(a, b)) # ゴキブリの生成方法は要検討
    X, Y = make_data(a * b, max(a, b))
    unit_positions = (X, Y)
    A = make_matrix(unit_positions, radius)
    return W, A

def solve_maximum_coverage_problem(facility_coverage: np.ndarray, p: int, w: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    最大カバー問題を解決する関数

    Parameters:
        facility_coverage (np.ndarray): 殺虫剤とゴキブリのカバー関係を示す行列
        p (int): 設置できる殺虫剤の最大数
        w (np.ndarray): 各地点におけるゴキブリの数

    Returns:
        Tuple[np.ndarray, float]: 選択された殺虫剤の配列と駆除されたゴキブリの合計数
    """
    num_facilities, num_users = facility_coverage.shape

    # 問題の定義
    prob = LpProblem("Maximum_Coverage", LpMaximize)

    # 変数の定義
    x = LpVariable.dicts("x", range(num_facilities), cat=LpBinary)
    z = LpVariable.dicts("z", range(num_users), cat=LpBinary)

    # 目的関数の設定: 駆除されたゴキブリの数を最大化
    prob += lpSum(w[i] * z[i] for i in range(num_users))

    # 制約条件: 選択する施設の数はp個以下
    prob += lpSum(x[j] for j in range(num_facilities)) == p

    # 制約条件: 各ゴキブリがカバーされる条件
    for i in range(num_users):
        prob += z[i] <= lpSum(facility_coverage[j, i] * x[j] for j in range(num_facilities))

    # 問題の解決
    prob.solve()

    # 結果の表示
    selected_facilities = np.array([x[j].varValue for j in range(num_facilities)])

    # 駆除されたゴキブリの合計数を計算
    total_killed = sum(w[i] * z[i].varValue for i in range(num_users))

    # 選択された施設のインデックスを表示
    selected_indices = [j for j in range(num_facilities) if selected_facilities[j] == 1]
    print("選択された殺虫剤のインデックス:", selected_indices)

    return selected_facilities, total_killed


def visualize_result(
    W: np.ndarray, A: np.ndarray, selected_facilities: np.ndarray, facility_positions: Tuple[np.ndarray, np.ndarray], radius: float
) -> None:
    """
    結果を可視化する関数

    Parameters:
        W (np.ndarray): ゴキブリの数を含む行列
        A (np.ndarray): 需要変数行列
        selected_facilities (np.ndarray): 選択された殺虫剤の配列
        facility_positions (Tuple[np.ndarray, np.ndarray]): 殺虫剤の座標
        radius (float): 殺虫剤の効果範囲
    """
    num_rows, num_cols = W.shape

    # グリッドの表示
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(-0.5, num_rows - 0.5)
    ax.set_xticks(np.arange(0, num_cols, 1))
    ax.set_yticks(np.arange(0, num_rows, 1))
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.7)

    # ゴキブリの数を表示
    for i in range(num_rows):
        for j in range(num_cols):
            size = min(W[i, j] * 10, 100)  # サイズを制限して視覚的に調整
            ax.plot(j, i, 'ro', markersize=size)
            ax.text(j, i, f"{W[i, j]}", ha="center", va="center", color="white", fontsize=10, weight='bold')

    # 殺虫剤の位置と影響範囲を表示
    for idx, (i, j) in enumerate(zip(*facility_positions)):
        if selected_facilities[idx] > 0:
            # 殺虫剤の位置を青色の円で表示
            circle_center = plt.Circle((j, i), 0.5, color='blue', fill=True, edgecolor='black', linewidth=1.5)
            ax.add_patch(circle_center)
            # 殺虫剤の影響範囲を青い円で表示（指定されたradiusを使用）
            influence_circle = plt.Circle((j, i), radius, color='blue', fill=False, linestyle='--', linewidth=2)
            ax.add_patch(influence_circle)

    # プロットを表示
    plt.gca().invert_yaxis()
    st.pyplot(fig)


# ストリームリットによるUIの作成
def main() -> None:
    """
    Streamlitを使った最大カバー問題のシミュレーションと可視化を行うメイン関数
    """
    st.title("最大カバー問題シミュレーション")
    st.write("ゴキブリの数と殺虫剤の設置場所を最適化します。")

    # Excelファイルアップロード
    uploaded_file = st.file_uploader("ゴキブリ分布のエクセルファイルをアップロードしてください", type=["xlsx"])
    if uploaded_file is not None:
        W = load_cockroach_distribution(uploaded_file)
    else:
        st.error("エクセルファイルをアップロードしてください。")
        return

    radius = st.slider("殺虫剤の効果範囲", min_value=0.1, max_value=10.0, value=0.1)
    p = st.slider("設置できる殺虫剤の最大数", min_value=1, max_value=20, value=5)

    if st.button("シミュレーション開始"):
        a, b = W.shape
        X, Y = make_data(a * b, max(a, b))
        facility_positions = (X, Y)
        A = make_matrix(facility_positions, radius)
        selected_facilities, total_killed = solve_maximum_coverage_problem(A, p, W.flatten())
        visualize_result(W, A, selected_facilities, facility_positions, radius)

        st.write(f"駆除されたゴキブリの合計数: {total_killed}")

if __name__ == "__main__":
    main()
