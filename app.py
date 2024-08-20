from typing import Any, Tuple

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import streamlit as st
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import Polygon


def load_cockroach_and_obstacles(file: "UploadedFile") -> np.ndarray[Any, Any]:
    """
    エクセルファイルからゴキブリの分布と障害物を読み込む関数

    Parameters:
        file: アップロードされたエクセルファイル

    Returns:
        np.ndarray: ゴキブリの数と障害物を含む行列
    """
    df = pd.read_excel(file, header=None)
    return df.values


def distance_with_obstacles(
    cockroach: Tuple[float, float],
    facility: Tuple[float, float],
    obstacles: np.ndarray
) -> float:
    """
    障害物を考慮したゴキブリと殺虫剤間の距離を計算する関数

    Parameters:
        cockroach (Tuple[float, float]): ゴキブリの位置 (x, y)
        facility (Tuple[float, float]): 殺虫剤の位置 (x, y)
        obstacles (np.ndarray): 障害物の位置を含む配列

    Returns:
        float: ゴキブリと殺虫剤間の障害物を考慮した距離
    """
    # ゴキブリと殺虫剤の位置を点として扱う
    start = Point(cockroach)
    end = Point(facility)

    # ゴキブリと殺虫剤の間の直線を定義
    line = LineString([start, end])

    # 障害物のポリゴンを作成
    obstacle_polygons = [Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]) for x, y in np.argwhere(obstacles)]

    # 障害物のポリゴンが直線と交差するかどうかを確認
    for polygon in obstacle_polygons:
        if line.intersects(polygon):
            return float('inf')  # 障害物が直線と交差する場合は距離を無限大に設定

    # 障害物がなければ直線距離を返す
    return line.length



def make_matrix_with_obstacles(
    cockroach_positions: Tuple[np.ndarray, np.ndarray],
    facility_positions: Tuple[np.ndarray, np.ndarray],
    W: np.ndarray,
    r: float
) -> np.ndarray:
    """
    障害物を考慮して需要変数行列を生成する関数

    Parameters:
        cockroach_positions (Tuple[np.ndarray, np.ndarray]): ゴキブリのx座標とy座標の配列
        facility_positions (Tuple[np.ndarray, np.ndarray]): 殺虫剤のx座標とy座標の配列
        W (np.ndarray): ゴキブリの数と障害物を含む行列
        r (float): 殺虫剤の効果範囲

    Returns:
        np.ndarray: 各殺虫剤が各ゴキブリをカバーするかどうかを示すバイナリ行列（0または1）
    """
    cockroach_list = np.array([cockroach_positions]).T
    facility_list = np.array([facility_positions]).T
    obstacles = W == -1

    def check(distance: float) -> int:
        """距離が効果範囲内かどうかを判定する関数"""
        return 1 if distance <= r else 0

    return np.array([
        [check(distance_with_obstacles(cockroach, facility, obstacles))
         for cockroach in cockroach_list]
        for facility in facility_list
    ])


def solve_maximum_coverage_problem(
    pest_coverage: np.ndarray,
    p: int,
    w: np.ndarray
) -> Tuple[np.ndarray, float]:
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
    selected_pos = np.array([x[j].value() for j in range(num_pest)])
    total_killed = problem.objective.value()

    return selected_pos, total_killed

def visualize_selected_facility_coverage(
    W: np.ndarray,
    A: np.ndarray,
    selected_pos: np.ndarray,
    facility_positions: Tuple[np.ndarray, np.ndarray],
    cockroach_positions: Tuple[np.ndarray, np.ndarray]
) -> None:
    """
    選択された殺虫剤とそのカバー範囲を可視化する関数

    Parameters:
        W (np.ndarray): ゴキブリの数と障害物を含む行列
        A (np.ndarray): ゴキブリの需要変数行列
        selected_pos (np.ndarray): 選択された殺虫剤のバイナリ配列
        facility_positions (Tuple[np.ndarray, np.ndarray]): 殺虫剤のx座標とy座標
        cockroach_positions (Tuple[np.ndarray, np.ndarray]): ゴキブリのx座標とy座標
    """
    obstacles = np.argwhere(W == -1)

    facility_x, facility_y = facility_positions
    selected_facility_x = np.array([facility_x[i] for i in range(len(facility_x)) if selected_pos[i] == 1], dtype=float)
    selected_facility_y = np.array([facility_y[i] for i in range(len(facility_y)) if selected_pos[i] == 1], dtype=float)

    grid_size = W.shape
    cell_size = 1

    fig, ax = plt.subplots(figsize=(12, 12))

    # グリッドの表示
    for x in range(grid_size[1]):
        for y in range(grid_size[0]):
            color = 'white'
            edge_color = 'black'
            if W[y, x] > 0:
                color = 'none'
            elif W[y, x] == 0:
                color = 'none'
            elif W[y, x] == -1:
                color = 'black'
            ax.add_patch(Rectangle((x, y), cell_size, cell_size, color=color, edgecolor=edge_color))

            if W[y, x] > 0:
                ax.text(x + 0.5, y + 0.5, str(W[y, x]), color='black', ha='center', va='center', fontsize=10,label='ゴキブリの数')

    # 選択された殺虫剤の表示
    for idx, (fx, fy) in enumerate(zip(facility_x, facility_y)):
        if selected_pos[idx] == 1:
            # idx番目の殺虫剤の位置をインデックスから取得
            ax.scatter(fx + 0.5, fy + 0.5, color='white', marker='x', s=100, label='殺虫剤の位置')

            # 駆除できるゴキブリを黄色で強調表示
            for j, (cy, cx) in enumerate(zip(cockroach_positions[0], cockroach_positions[1])):
                if A[idx, j] == 1:
                    ax.add_patch(Rectangle((cx, cy), cell_size, cell_size, color='yellow', alpha=0.5, edgecolor='black'))

    # 障害物を目立たせるための赤い枠線
    for obs in obstacles:
        ax.add_patch(Rectangle((obs[1], obs[0]), cell_size, cell_size, color='none', edgecolor='red', linestyle='--'))

    # 凡例の設定
    handles = [
        Rectangle((0, 0), 1, 1, color='yellow', alpha=0.5),
        Rectangle((0, 0), 1, 1, color='black', edgecolor='black', linestyle='--', linewidth=2),
        Rectangle((0, 0), 1, 1, color='white', edgecolor='black', linestyle='-', linewidth=1)
    ]

    labels = [
        '駆除できるゴキブリ',
        '障害物',
        "ゴキブリの数"
    ]
    ax.legend(handles=handles, labels=labels, loc='best')
    ax.set_xlabel('X 座標')
    ax.set_ylabel('Y 座標')
    ax.set_title('選択された殺虫剤と駆除できるゴキブリ')
    ax.grid(True)
    ax.invert_yaxis()

    # Streamlitに描画
    st.pyplot(fig)




def main() -> None:
    """
    Streamlit アプリケーションのメイン関数
    """
    st.title("最大カバー問題のシミュレーション - ゴキブリ駆除の最適配置")
    st.sidebar.title("設定")

    # エクセルファイルのアップロード
    uploaded_file = st.sidebar.file_uploader("ゴキブリの分布データをアップロードしてください", type=["xlsx"])

    if uploaded_file is not None:
        W = load_cockroach_and_obstacles(uploaded_file)
        st.sidebar.success("ファイルのアップロードに成功しました")

        # パラメータの設定
        p = st.sidebar.number_input("設置可能な殺虫剤の数", min_value=1, max_value=50, value=3)
        r = st.sidebar.number_input("殺虫剤の効果範囲", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

        # ゴキブリと施設の座標
        cockroach_positions = np.where(W >= 0)
        facility_positions = np.where(W != -1)

        # 需要変数行列の作成
        A = make_matrix_with_obstacles(cockroach_positions, facility_positions, W, r)

        # 最大カバー問題を解く
        selected_pos, total_killed = solve_maximum_coverage_problem(A, p, W[cockroach_positions])

        st.write(f"駆除されたゴキブリの合計数: {total_killed}")

        # 結果の可視化
        visualize_selected_facility_coverage(W, A, selected_pos, facility_positions, cockroach_positions)


if __name__ == "__main__":
    main()
