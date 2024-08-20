from typing import Any, Tuple
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import streamlit as st
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

# ゴキブリの分布をエクセルファイルから読み込む関数
def load_cockroach_distribution(file: "UploadedFile") -> np.ndarray[Any, Any]:
    """
    エクセルファイルからゴキブリの分布を読み込む関数

    Parameters:
        file: アップロードされたエクセルファイル

    Returns:
        np.ndarray: ゴキブリの数を含む行列
    """
    df = pd.read_excel(file, header=None)
    return df.values

# ゴキブリと殺虫剤の距離を測る
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    2点間のユークリッド距離を計算する関数

    Parameters:
        p1 (np.ndarray): 1つ目の点の座標を含む配列 (x, y) の形
        p2 (np.ndarray): 2つ目の点の座標を含む配列 (x, y) の形

    Returns:
        float: 2点間のユークリッド距離
    """
    return np.linalg.norm(p1 - p2)

# 需要変数行列を生成する関数
def make_matrix(
    cockroach_positions: Tuple[np.ndarray, np.ndarray],
    facility_positions: Tuple[np.ndarray, np.ndarray],
    r: float
) -> np.ndarray:
    """
    ゴキブリと殺虫剤の位置に基づいて需要変数行列を生成する関数

    Parameters:
        cockroach_positions (Tuple[np.ndarray, np.ndarray]): ゴキブリのx座標とy座標の配列
        facility_positions (Tuple[np.ndarray, np.ndarray]): 殺虫剤のx座標とy座標の配列
        r (float): 殺虫剤の効果範囲

    Returns:
        np.ndarray: 各殺虫剤が各ゴキブリをカバーするかどうかを示すバイナリ行列（0または1）
    """
    cockroach_list = np.array([cockroach_positions]).T
    facility_list = np.array([facility_positions]).T

    def check(a: float) -> int:
        """距離が効果範囲内かどうかを判定する関数"""
        return 1 if a <= r else 0

    return np.array([[check(distance(cockroach, facility)) for cockroach in cockroach_list] for facility in facility_list])

# パラメータを生成する関数
def generate_parameters_from_file(
    W: np.ndarray, facility_positions: Tuple[np.ndarray, np.ndarray], radius: float
) -> np.ndarray:
    """
    ゴキブリの分布に基づいて需要変数の行列を生成する関数

    Parameters:
        W (np.ndarray): ゴキブリの数を含む行列
        facility_positions (Tuple[np.ndarray, np.ndarray]): 殺虫剤の位置
        radius (float): 殺虫剤の効果範囲

    Returns:
        np.ndarray: 需要変数行列A
    """
    num_rows, num_cols = W.shape
    X, Y = np.indices((num_rows, num_cols))
    cockroach_positions = (X.flatten(), Y.flatten())
    A = make_matrix(cockroach_positions, facility_positions, radius)
    return A

# 最大カバー問題を解決する関数
def solve_maximum_coverage_problem(pest_coverage: np.ndarray, p: int, w: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    最大カバー問題を解決する関数

    Parameters:
        pest_coverage (np.ndarray): 殺虫剤とゴキブリのカバー関係を示す行列
        p (int): 設置できる殺虫剤の最大数
        w (np.ndarray): 各地点におけるゴキブリの数

    Returns:
        Tuple[np.ndarray, float]: 選択された殺虫剤の配列と駆除されたゴキブリの合計数
    """
    num_pest, num_g = pest_coverage.shape

    # 問題の定義
    problem = pulp.LpProblem("殺虫剤の配置最適化", pulp.LpMaximize)

    # 変数の定義
    x = pulp.LpVariable.dicts("x", range(num_pest), cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("z", range(num_g), cat=pulp.LpBinary)

    # 目的関数の設定: 駆除されたゴキブリの数を最大化
    problem += pulp.lpSum(w[i] * z[i] for i in range(num_g))

    # 制約条件: 選択する施設の数はp個以下
    problem += pulp.lpSum(x[j] for j in range(num_pest)) <= p

    # 制約条件: 各ゴキブリがカバーされる条件
    for i in range(num_g):
        problem += z[i] <= pulp.lpSum(pest_coverage[j, i] * x[j] for j in range(num_pest))

    # 問題の解決
    problem.solve()

    # 結果の表示
    selected_pos = np.array([x[j].varValue for j in range(num_pest)])

    # 駆除されたゴキブリの合計数を計算
    total_killed = problem.objective.value()

    return selected_pos, total_killed

# 結果を可視化する関数
def visualize_result(
    W: np.ndarray, A: np.ndarray, selected_pos: np.ndarray, pest_positions: Tuple[np.ndarray, np.ndarray], radius: float
) -> None:
    """
    結果を可視化する関数

    Parameters:
        W (np.ndarray): ゴキブリの数を含む行列
        A (np.ndarray): 需要変数行列
        selected_pos (np.ndarray): 選択された殺虫剤の配列
        pest_positions (Tuple[np.ndarray, np.ndarray]): 殺虫剤の座標
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
    for idx, (i, j) in enumerate(zip(*pest_positions)):
        if selected_pos[idx] > 0:
            # 殺虫剤の位置を青色の円で表示
            circle_center = plt.Circle((j, i), 0.5, color='blue', fill=True, edgecolor='black', linewidth=1.5)
            ax.add_patch(circle_center)
            # 殺虫剤の影響範囲を青い円で表示
            influence_circle = plt.Circle((j, i), radius, color='blue', fill=False, linestyle='--', linewidth=2)
            ax.add_patch(influence_circle)

    # プロットを表示
    plt.gca().invert_yaxis()
    st.pyplot(fig)

# ストリームリットによるUIの構築
def main() -> None:
    st.title("最大カバー問題のシミュレーション - ゴキブリ駆除の最適化")
    st.sidebar.title("設定")

    # エクセルファイルのアップロード
    uploaded_file = st.sidebar.file_uploader("ゴキブリの分布データをアップロードしてください", type=["xlsx"])

    if uploaded_file:
        W = load_cockroach_distribution(uploaded_file)
        st.sidebar.success("ファイルのアップロードに成功しました")

        # 殺虫剤の効果範囲と配置可能数の入力
        radius = st.sidebar.number_input("殺虫剤の効果範囲", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        p = st.sidebar.number_input("設置可能な殺虫剤の数", min_value=1, max_value=50, value=3)

        # 需要変数行列の生成
        num_rows, num_cols = W.shape
        X, Y = np.indices((num_rows, num_cols))
        pest_positions = (X.flatten(), Y.flatten())
        A = generate_parameters_from_file(W, pest_positions, radius)

        # 最大カバー問題の解決
        selected_pos, total_killed = solve_maximum_coverage_problem(A, p, W.flatten())

        st.sidebar.write(f"駆除されたゴキブリの合計数: {int(total_killed)}")

        # 結果の可視化
        visualize_result(W, A, selected_pos, pest_positions, radius)

if __name__ == "__main__":
    main()
