import io
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit as st


@st.cache_data
def generate_parameters(
    a: int, b: int, G: List[str], positions: Dict[str, Tuple[int, int]], seed: int = 0
) -> Tuple[
    Dict[str, int],
    Dict[Tuple[str, int, int, int, int], float],
    Dict[Tuple[int, int, int, int], float],
]:
    np.random.seed(seed)

    # N_gの生成
    N_g = {g: np.random.randint(5, 10) for g in G}

    # V_gの生成
    V_g = {}
    for g in G:
        i_pos, j_pos = positions[g]
        for i in range(a):
            for j in range(b):
                distance = np.sqrt((i - i_pos) ** 2 + (j - j_pos) ** 2)
                V_g[(g, i_pos, j_pos, i, j)] = np.exp(-distance)

    # Dの生成
    D = {}
    for i in range(a):
        for j in range(b):
            for i_prime in range(a):
                for j_prime in range(b):
                    distance = np.sqrt((i - i_prime) ** 2 + (j - j_prime) ** 2)
                    D[(i, j, i_prime, j_prime)] = np.exp(-distance)

    return N_g, V_g, D


def solve_cockroach_elimination(
    a: int,
    b: int,
    G: List[str],
    positions: Dict[str, Tuple[int, int]],
    N_g: Dict[str, int],
    V_g: Dict[Tuple[str, int, int, int, int], float],
    D: Dict[Tuple[int, int, int, int], float],
    max_pesticides: int,
) -> Dict[Tuple[int, int], int]:
    # 決定変数
    W = pulp.LpVariable.dicts(
        "W", [(i, j) for i in range(a) for j in range(b)], 0, 1, cat=pulp.LpBinary
    )

    # 問題の定式化
    prob = pulp.LpProblem("CockroachElimination", pulp.LpMaximize)

    # 目的関数の設定
    # V_g(i_pos,j_pos,i',j') * D(i,j,i',j')のi',j'についての積を計算
    obj_1_dict = {}
    for g in G:
        i_pos, j_pos = positions[g]
        for i in range(a):
            for j in range(b):
                obj_1 = 1.0
                for i_prime in range(a):
                    for j_prime in range(b):
                        obj_1 *= V_g[(g, i_pos, j_pos, i_prime, j_prime)] * D[(i, j, i_prime, j_prime)]
                obj_1_dict[(g, i, j)] = obj_1

    # pulp.lpSum(N_g[g] * pulp.lpSum([W[(i, j)] * obj_1_dict[(g, i, j, i_prime, j_prime)] for i in range(a) for j in range(b) for i_prime in range(a) for j_prime in range(b)]) for g in G)を計算
    prob += pulp.lpSum(
        [
            N_g[g]
            * pulp.lpSum(
                [W[(i, j)] * obj_1_dict[(g, i, j)] for i in range(a) for j in range(b)]
            )
            for g in G
        ]
    )


    # 制約条件の追加
    prob += pulp.lpSum([W[(i, j)] for i in range(a) for j in range(b)]) <= max_pesticides

    # 問題を解く
    prob.solve()

    # 結果を取得
    result = {(i, j): int(W[(i, j)].varValue) for i in range(a) for j in range(b)}

    # 結果を表示
    print(pulp.LpStatus[prob.status])

    return result


def visualize_result(
    a: int,
    b: int,
    result: Dict[Tuple[int, int], int],
    positions: Dict[str, Tuple[int, int]],
    N_g: Dict[str, int],
) -> io.BytesIO:
    """
    結果を可視化する関数

    Parameters:
        a (int): グリッドの行数
        b (int): グリッドの列数
        result (dict): 各セルに設置される殺虫剤の結果を含む辞書
            - (i, j): セル (i, j) における殺虫剤の設置結果 (0 または 1)
        positions (dict): 各群れの初期位置
        N_g (dict): 各群れに属するゴキブリの数

    Returns:
        io.BytesIO: 生成されたグラフ画像を含むバッファ
    """
    fig, ax = plt.subplots()

    # グリッドを描画
    for i in range(a):
        for j in range(b):
            if result[(i, j)] == 1:
                ax.add_patch(plt.Rectangle((j, a - i - 1), 1, 1, color="red"))

    # ゴキブリの群れの位置と大きさを描画
    for g, pos in positions.items():
        i_pos, j_pos = pos
        size = N_g[g]
        circle = plt.Circle((j_pos + 0.5, a - i_pos - 0.5), size / 10, color="blue", alpha=0.5)
        ax.add_patch(circle)
        ax.text(j_pos + 0.5, a - i_pos - 0.5, f"{size}", color="white", ha="center", va="center")

    ax.set_xlim(0, b)
    ax.set_ylim(0, a)
    ax.set_aspect("equal", adjustable="box")

    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return buf


def main() -> None:
    """
    Streamlit アプリケーションのメイン関数
    """
    # タイトル
    st.title("ゴキブリの群れの撲滅問題")

    # グリッドの行数と列数
    a: int = st.slider("グリッドの行数", 5, 20, 10)
    b: int = st.slider("グリッドの列数", 5, 20, 10)

    # ゴキブリの群れのリスト
    G: List[str] = ["A", "B", "C"]

    # 各群れの初期位置
    positions: Dict[str, Tuple[int, int]] = {
        "A": (np.random.randint(0, a), np.random.randint(0, b)),
        "B": (np.random.randint(0, a), np.random.randint(0, b)),
        "C": (np.random.randint(0, a), np.random.randint(0, b)),
    }

    # パラメータの生成
    N_g, V_g, D = generate_parameters(a, b, G, positions)

    # 殺虫剤の最大設置数
    max_pesticides: int = st.slider("殺虫剤の最大設置数", 1, 10, 5)

    # ゴキブリの群れの撲滅問題を解く
    result: Dict[Tuple[int, int], int] = solve_cockroach_elimination(
        a, b, G, positions, N_g, V_g, D, max_pesticides
    )

    # 結果の可視化
    buf: io.BytesIO = visualize_result(a, b, result, positions, N_g)
    st.image(buf)

    # W_ijの値を表示
    st.write("W_ijの値")

    for i in range(a):
        for j in range(b):
            st.write(f"W[{i}, {j}] = {result[(i, j)]}")

    # ゴキブリの群れの数を表示
    st.write("ゴキブリの群れの数")
    st.write(N_g)

    # ゴキブリの移動確率を表示
    st.write("ゴキブリの移動確率")
    st.write(V_g)

    # 殺虫剤に誘引される確率を表示
    st.write("殺虫剤に誘引される確率")
    st.write(D)


if __name__ == "__main__":
    main()
