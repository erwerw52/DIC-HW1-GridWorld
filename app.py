import streamlit as st
import random

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


def trace_path(start, goal, policy, n, obstacles):
    """沿著策略從起點追蹤至終點，回傳路徑格子集合（含起終點）。"""
    path = []
    cell = start
    visited = set()
    max_steps = n * n
    while cell != goal and len(path) < max_steps:
        if cell in visited:
            break  # 偵測到迴圈，中止
        visited.add(cell)
        path.append(cell)
        i, j = cell
        action = policy[i][j]
        if action is None:
            break
        di, dj = ACTIONS[action]
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
            cell = (ni, nj)
        else:
            break
    if cell == goal:
        path.append(goal)
    return set(path)


def init_state(n):
    st.session_state.grid_n = n
    st.session_state.start = None
    st.session_state.goal = None
    st.session_state.obstacles = set()
    st.session_state.strategy = {}
    st.session_state.V = None
    st.session_state.policy = None
    st.session_state.path = set()


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
        st.session_state.path = set()
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

    /* ── 壓縮 Streamlit 預設頂部 padding，讓內容不需捲動 ── */
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* ── 標題縮緊邊距 ── */
    h1 { margin-top: 0 !important; margin-bottom: 0.15rem !important; font-size: 1.7rem !important; }
    h5 { margin-top: 0 !important; margin-bottom: 0.2rem !important; }
    
    /* ── 網格格子按鈕：正方形、草地風格 ── */
    [class*="st-key-cell_"] button {
        aspect-ratio: 1 / 1 !important;
        width: 90% !important;
        height: 80px !important;
        font-size: 30px !important;
        font-weight: bold;
        border-radius: 14px;
        border: 2px solid #a3c9a8;
        background-color: #ffffff;
        color: #4a5d23;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        box-shadow: 2px 2px 6px rgba(100,140,80,0.12);
        line-height: 1.1;
    }
    [class*="st-key-cell_"] button p {
        font-size: 24px !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        color: inherit !important;
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
        height: 44px !important;
        font-size: 15px !important;
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
        padding: 0px 0px;
    }
    
    /* 調整垂直間隔 */
    [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }
    
    /* 標題與文字顏色設定 */
    h1, h2, h3, h4, h5, h6, p {
        color: #5c4a3d !important;
    }

    /* ── 右側玩法說明面板 ── */
    .guide-panel {
        background: linear-gradient(160deg, #fffdf5 0%, #f5f0e8 100%);
        border: 2px solid #c8dfc4;
        border-radius: 18px;
        padding: 20px 18px 24px 18px;
        box-shadow: 3px 4px 16px rgba(120,160,100,0.13);
        font-family: inherit;
    }
    .guide-panel-title {
        font-size: 18px;
        font-weight: 800;
        color: #4a5d23 !important;
        letter-spacing: 0.04em;
        margin-bottom: 14px;
        border-bottom: 2px dashed #c8dfc4;
        padding-bottom: 8px;
    }
    .guide-section {
        font-size: 13px;
        font-weight: 700;
        color: #7a5c3a !important;
        margin: 14px 0 8px 0;
        letter-spacing: 0.05em;
    }
    .guide-step {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 10px;
    }
    .step-num {
        min-width: 24px;
        height: 24px;
        background-color: #5a8a50;
        color: #fff;
        border-radius: 50%;
        font-size: 12px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .step-text {
        font-size: 13px;
        color: #5c4a3d !important;
        line-height: 1.6;
    }
    .guide-divider {
        border: none;
        border-top: 1px dashed #d4c9b8;
        margin: 14px 0;
    }
    .legend-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 7px;
        font-size: 13px;
        color: #5c4a3d !important;
    }
    .legend-icon {
        font-size: 18px;
        width: 26px;
        text-align: center;
    }
    .path-badge {
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: #7cfc00;
        border: 2px solid #081c15;
        border-radius: 4px;
        vertical-align: middle;
    }
    </style>
    """

def build_path_css(path):
    """為路徑中的每個格子動態生成綠色背景 CSS，確保與白色文字對比良好。"""
    if not path:
        return ""
    rules = []
    for (i, j) in path:
        rules.append(f"""
    [class*="st-key-cell_{i}_{j}"] button {{
        background-color: #7cfc00 !important;
        border-color: #081c15 !important;
        color: #1a3a00 !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
        box-shadow: 0 0 14px rgba(27,67,50,0.65) !important;
    }}
    [class*="st-key-cell_{i}_{j}"] button p {{
        color: #1a3a00 !important;
    }}
    [class*="st-key-cell_{i}_{j}"] button:hover {{
        background-color: #2d6a4f !important;
        border-color: #081c15 !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(45,106,79,0.8) !important;
    }}
    [class*="st-key-cell_{i}_{j}"] button:hover p {{
        color: #ffffff !important;
    }}""")
    return f"<style>{''.join(rules)}</style>"


def main():
    st.set_page_config(page_title="Happy Farm Strategy", page_icon="🌻", layout="wide")
    st.markdown(build_css(), unsafe_allow_html=True)
    
    st.title("🚜 歡樂農場尋寶記 🌻")
    st.markdown("##### 🐶 找出回到溫暖小屋的最佳路徑！")

    # ── 主要雙欄佈局：左側農場（4）、右側玩法說明（1） ──
    col_main, col_guide = st.columns([3, 1], gap="large")

    with col_main:
        n = st.selectbox("▸ 🌾 請選擇你的農場大小 n（3–9）", list(range(3, 10)), index=0)

        if 'grid_n' not in st.session_state or st.session_state.grid_n != n:
            init_state(n)

        # 注入路徑格子的動態 CSS（需在 init_state 之後，確保 path 已存在）
        if 'path' not in st.session_state:
            st.session_state.path = set()
        st.markdown(build_path_css(st.session_state.path), unsafe_allow_html=True)

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
        grid_container = st.container()
        with grid_container:
            for i in range(n):
                cols = st.columns(n, gap="small")
                for j in range(n):
                    with cols[j]:
                        arrow = st.session_state.strategy.get((i, j), '?')
                        if (i, j) == st.session_state.start:
                            label = "🚜"
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
                st.session_state.path = trace_path(
                    st.session_state.start,
                    st.session_state.goal,
                    policy,
                    n,
                    st.session_state.obstacles,
                )
                st.rerun()

        if st.session_state.V is not None:
            st.success("🎉 太棒了！已經幫你找好回家的路囉！")

    # ── 右側玩法說明面板 ──
    with col_guide:
        st.markdown("""
<div class="guide-panel">
  <div class="guide-panel-title">📖 遊戲玩法</div>

  <div class="guide-section">⚙️ 設定地圖</div>

  <div class="guide-step">
    <span class="step-num">1</span>
    <span class="step-text">點擊任意空地<br>設定 🚜 <b>起點</b></span>
  </div>
  <div class="guide-step">
    <span class="step-num">2</span>
    <span class="step-text">點擊另一格空地<br>設定 🏡 <b>終點</b></span>
  </div>
  <div class="guide-step">
    <span class="step-num">3</span>
    <span class="step-text">繼續點其他空地<br>種下 🌳 <b>障礙物</b><br><small>（最多 n−2 棵）</small></span>
  </div>
  <div class="guide-step">
    <span class="step-num">4</span>
    <span class="step-text">再次點擊已設格子<br>可 <b>取消</b> 該設定</span>
  </div>

  <hr class="guide-divider">
  <div class="guide-section">🧭 開始導航</div>

  <div class="guide-step">
    <span class="step-num">5</span>
    <span class="step-text">按下<br>「✨ 開始導航！」<br>計算最佳策略</span>
  </div>
  <div class="guide-step">
    <span class="step-num">6</span>
    <span class="step-text">每格顯示<br><b>箭頭</b>（方向）<br><b>數字</b>（預期回報）</span>
  </div>
  <div class="guide-step">
    <span class="step-num">7</span>
    <span class="step-text">沿綠色格子走<br>即是 <b>最短路徑</b>！</span>
  </div>

  <hr class="guide-divider">
  <div class="guide-section">🗺️ 圖例</div>

  <div class="legend-row"><span class="legend-icon">🚜</span> 牽引機（起點）</div>
  <div class="legend-row"><span class="legend-icon">🏡</span> 小屋（終點）</div>
  <div class="legend-row"><span class="legend-icon">🌳</span> 樹木（障礙物）</div>
  <div class="legend-row"><span class="legend-icon">🌾</span> 農田（可通行）</div>
  <div class="legend-row"><span class="legend-icon">↑↓←→</span> 最佳方向</div>
  <div class="legend-row"><span class="legend-icon">−X.X</span> 預期回報值</div>
  <div class="legend-row"><span class="legend-icon"><span class="path-badge"></span></span> 最佳路徑格</div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
