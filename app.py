import streamlit as st
import random
import base64
from pathlib import Path

ARROWS = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

def value_iteration(n, goal, obstacles, gamma=0.9, theta=1e-4):
    """執行 Value Iteration，回傳 value 矩陣與最佳策略。"""
    V = [[0.0] * n for _ in range(n)]
    policy = [[None] * n for _ in range(n)]

    def is_free(i, j):
        return 0 <= i < n and 0 <= j < n and (i, j) not in obstacles

    def next_state(i, j, a):
        di, dj = ACTIONS[a]
        ni, nj = i + di, j + dj
        return (ni, nj) if is_free(ni, nj) else (i, j)

    while True:
        delta = 0.0
        for i in range(n):
            for j in range(n):
                if (i, j) in obstacles or (i, j) == goal:
                    continue
                old_v = V[i][j]
                best_v = float('-inf')
                best_a = None
                for a in ACTIONS:
                    ni, nj = next_state(i, j, a)
                    reward = 0 if (ni, nj) == goal else -1
                    val = reward + gamma * V[ni][nj]
                    if val > best_v:
                        best_v = val
                        best_a = a
                V[i][j] = best_v
                policy[i][j] = best_a
                delta = max(delta, abs(old_v - best_v))
        if delta < theta:
            break

    return V, policy


def init_state(n):
    st.session_state.grid_n = n
    st.session_state.start = None
    st.session_state.goal = None
    st.session_state.obstacles = set()
    st.session_state.strategy = {}
    st.session_state.V = None
    st.session_state.policy = None


def handle_click(i, j, max_obs):
    """處理點擊格子的狀態變化，使用 Callback 避免二次觸發造成延遲 (rerun twice)"""
    cell = (i, j)
    used_obs = len(st.session_state.obstacles)
    changed = True
    
    if st.session_state.start is None:
        st.session_state.start = cell
    elif st.session_state.goal is None and cell != st.session_state.start:
        st.session_state.goal = cell
    else:
        if cell == st.session_state.start:
            st.session_state.start = None
        elif cell == st.session_state.goal:
            st.session_state.goal = None
        else:
            if cell in st.session_state.obstacles:
                st.session_state.obstacles.discard(cell)
            elif used_obs < max_obs:
                st.session_state.obstacles.add(cell)
            else:
                changed = False
                
    if changed:
        # 重置價值計算與策略
        st.session_state.V = None
        st.session_state.policy = None
        st.session_state.strategy = {}


def build_css():
    return """
    <style>
    /* 全局背景色與字體顏色調整為農場溫暖風格 */
    [data-testid="stAppViewContainer"] {
        background-color: #fdfbf7;
    }
    .stApp {
        background-color: #fdfbf7;
    }
    
    /* ── 網格格子按鈕：正方形、草地風格 ── */
    [class*="st-key-cell_"] button {
        aspect-ratio: 1 / 1 !important;
        width: 100% !important;
        height: auto !important;
        font-size: 22px;
        font-weight: bold;
        border-radius: 14px;
        border: 2px solid #a3c9a8;
        background-color: #ffffff;
        color: #4a5d23;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        box-shadow: 2px 2px 6px rgba(100,140,80,0.12);
        line-height: 1.4;
    }
    [class*="st-key-cell_"] button:hover {
        transform: scale(1.08) rotate(1deg);
        border-color: #ffb703;
        background-color: #fff8e1;
        box-shadow: 3px 3px 12px rgba(200,140,0,0.2);
        color: #b07800;
    }

    /* ── 導航計算按鈕：橫幅、同色系綠金農場風格 ── */
    [class*="st-key-calc_btn"] button {
        width: 100% !important;
        height: 52px !important;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        letter-spacing: 0.05em;
        background-color: #ffffff !important;
        border: 2px solid #5a8a50 !important;
        color: #4a5d23 !important;
        transition: background-color 0.2s, transform 0.1s, box-shadow 0.2s;
    }
    [class*="st-key-calc_btn"] button:hover {
        background-color: #f4f9f4 !important;
        border-color: #4a7040 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(74, 112, 64, 0.35);
    }
    
    /* 調整列之間的間距以讓網格更緊湊 */
    [data-testid="column"] {
        padding: 0px 4px;
    }
    
    /* 調整垂直間隔，讓它與水平 gap="small" 的間距接近 */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* 標題與文字顏色設定 */
    h1, h2, h3, h4, h5, h6, p {
        color: #5c4a3d !important;
    }
    </style>
    """

def main():
    st.set_page_config(page_title="Happy Farm Strategy", page_icon="🌻", layout="centered")
    st.markdown(build_css(), unsafe_allow_html=True)
    
    st.title("🚜 歡樂農場尋寶記 🌻")
    st.markdown("##### 🐶 找出回到溫暖小屋的最佳路徑！")

    n = st.selectbox("▸ 🌾 請選擇你的農場大小 n（3–9）", list(range(3, 10)), index=0)

    if 'grid_n' not in st.session_state or st.session_state.grid_n != n:
        init_state(n)

    max_obs = n - 2
    used_obs = len(st.session_state.obstacles)
    remaining = max_obs - used_obs

    st.info(
        f"🎮 **農場小提示**：依序點擊空地設定 **起點 (🚜牽引機)** → **終點 (🏡小屋)** → **障礙物 (🌳樹木)**。\n\n"
        f"🔲 **樹木種金額度**：還可以種植 {remaining} 棵 （總共最多 {max_obs} 棵）"
    )

    # ── 確保每格都有隨機策略箭頭 ──
    for i in range(n):
        for j in range(n):
            if (i, j) not in st.session_state.strategy:
                st.session_state.strategy[(i, j)] = random.choice(list(ARROWS.values()))

    # ── 繪製 n×n 格子 ──
    # 在外層放一個 container 讓它具有獨立背景
    grid_container = st.container()
    with grid_container:
        for i in range(n):
            cols = st.columns(n, gap="small")
            for j in range(n):
                with cols[j]:
                    arrow = st.session_state.strategy.get((i, j), '?')
                    if (i, j) == st.session_state.start:
                        label = f"🚜"
                    elif (i, j) == st.session_state.goal:
                        label = '🏡'
                    elif (i, j) in st.session_state.obstacles:
                        label = '🌳'
                    else:
                        if st.session_state.V is not None:
                            v_val = st.session_state.V[i][j]
                            p_dir = st.session_state.policy[i][j]
                            label = f"{ARROWS.get(p_dir,'')}\n{v_val:.2f}"
                        else:
                            label = f"🌾\n{arrow}"

                    # 使用 on_click 觸發邏輯，能讓畫面重繪更為即時
                    st.button(label, key=f"cell_{i}_{j}", on_click=handle_click, args=(i, j, max_obs), use_container_width=True)

    st.divider()

    # 計算按鈕
    calc = st.button("✨ 開始導航！(Value Iteration)", key="calc_btn", type="primary", use_container_width=True)

    if calc:
        if st.session_state.start is None or st.session_state.goal is None:
            st.error("❌ 哎呀！請先設定好牽引機 (起點) 與小屋 (終點) 喔！")
        elif st.session_state.start == st.session_state.goal:
            st.error("❌ 起點與終點不能在同一個地方啦！")
        else:
            V, policy = value_iteration(
                n,
                st.session_state.goal,
                st.session_state.obstacles,
            )
            st.session_state.V = V
            st.session_state.policy = policy
            st.rerun()

    if st.session_state.V is not None:
        st.success("🎉 太棒了！已經幫你找好回家的路囉！")


if __name__ == "__main__":
    main()
