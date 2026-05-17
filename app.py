import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import os

# 解決 matplotlib 中文亂碼問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="台積電2330多功能交易回測系統", layout="wide")
st.title("📊 台積電 2330 - 完整量化交易回測與 AI 評估系統 (策略一至六)")

# ==================== 1. 核心回測引擎 (時空精準對齊版) ====================
class AssignmentRecord:
    def __init__(self, total_bars):
        self.OpenInterestQty = 0
        self.OrderPrice = 0
        self.Profit = []
        self.TotalProfit = 0
        self.WinCount = 0
        self.TotalCount = 0
        # 讓資產歷史從第一天開始，長度跟 K 線完全一致
        self.EquityHistory = np.zeros(total_bars)
        self.MaxEquity = 0
        self.MDD = 0

    def Order(self, BS, OrderPrice, OrderQty=3):
        if self.OpenInterestQty == 0:
            self.OrderPrice = OrderPrice
            self.OpenInterestQty = OrderQty if (BS == 'B' or BS == 'Buy') else -OrderQty

    def Cover(self, BS, OrderPrice, current_idx):
        if self.OpenInterestQty != 0:
            if self.OpenInterestQty > 0 and (BS == 'S' or BS == 'Sell'):
                profit = (OrderPrice - self.OrderPrice) * self.OpenInterestQty * 1000
            elif self.OpenInterestQty < 0 and (BS == 'B' or BS == 'Buy'):
                profit = (self.OrderPrice - OrderPrice) * (-self.OpenInterestQty) * 1000
            else:
                return
            
            self.Profit.append(profit)
            self.TotalProfit += profit
            self.TotalCount += 1
            if profit > 0:
                self.WinCount += 1
                
            # 精準記錄「這根 K 線當下」的總盈虧
            self.EquityHistory[current_idx] = self.TotalProfit
            
            if self.TotalProfit > self.MaxEquity:
                self.MaxEquity = self.TotalProfit
            mdd = self.MaxEquity - self.TotalProfit
            if mdd > self.MDD:
                self.MDD = mdd
                
            self.OpenInterestQty = 0

    def FillRemainingEquity(self):
        """ 讓沒有交易發生的 K 線，承接上一次的資產餘額 """
        current_balance = 0
        for i in range(len(self.EquityHistory)):
            if self.EquityHistory[i] == 0 and i > 0:
                self.EquityHistory[i] = current_balance
            elif self.EquityHistory[i] != 0:
                current_balance = self.EquityHistory[i]

    def GetWinRate(self):
        return self.WinCount / self.TotalCount if self.TotalCount > 0 else 0

# ==================== 2. 資料庫讀取 ====================
@st.cache_data
def load_and_process_data():
    db_path = "shioaji.db"
    if not os.path.exists(db_path):
        db_path = r"C:\Users\Minir\OneDrive\桌面\量化交易期末報告\shioaji.db"
    if not os.path.exists(db_path):
        st.error(f"❌ 找不到 shioaji.db！")
        st.stop()
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    target_table = "stock_KBar_2330" if "stock_KBar_2330" in tables else tables[0]
    
    df = pd.read_sql_query(f"SELECT * FROM {target_table}", conn)
    conn.close()
    
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # 採樣成小時線並去除無資料期間
    df_hourly = df.resample('60min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    
    return df_hourly, db_path

df_hourly, successful_db_path = load_and_process_data()
st.caption(f"💡 目前成功連線資料庫：{successful_db_path} (總資料筆數: {len(df_hourly)} 筆)")

# ==================== 3. 側邊欄控制面板 ====================
st.sidebar.header("⚙️ 策略與參數控制面板")
strategy_choice = st.sidebar.selectbox(
    "選擇交易策略",
    ["(一) 移動平均策略 (MA)", "(二) RSI 順勢策略", "(三) RSI 逆勢策略", "(四) 布林通道策略 (BBands)", "(五) MACD 趨勢策略", "(六) KDJ 震盪策略"]
)

# 簡化參數初始化 (維持原有天期)
p1, p2, p3, stop_loss = 20, 5, 0, 10
if strategy_choice == "(一) 移動平均策略 (MA)":
    p1 = st.sidebar.slider("長天期均線 (Long MA)", 20, 60, 20)
    p2 = st.sidebar.slider("短天期均線 (Short MA)", 5, 19, 5)
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, 10)
elif strategy_choice == "(二) RSI 順勢策略":
    p1 = st.sidebar.slider("RSI 週期", 5, 30, 14)
    p2 = st.sidebar.slider("順勢買入超買界線", 50, 80, 50)
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, 20)
elif strategy_choice == "(三) RSI 逆勢策略":
    p1 = st.sidebar.slider("RSI 週期", 5, 30, 14)
    p2 = st.sidebar.slider("逆勢買入低估界線", 10, 45, 30)
    p3 = st.sidebar.slider("逆勢賣出高估界線", 55, 90, 70)
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, 15)
elif strategy_choice == "(四) 布林通道策略 (BBands)":
    p1 = st.sidebar.slider("中線週期 (MA period)", 5, 40, 20)
    p2 = st.sidebar.slider("標準差倍數 (Std Dev)", 1, 3, 2)
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, 25)
elif strategy_choice == "(五) MACD 趨勢策略":
    p1 = st.sidebar.slider("MACD 快線週期", 5, 20, 12)
    p2 = st.sidebar.slider("MACD 慢線週期", 21, 40, 26)
    p3 = st.sidebar.slider("訊號線週期", 5, 15, 9)
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, 30)
elif strategy_choice == "(六) KDJ 震盪策略":
    p1 = st.sidebar.slider("KDJ 快線週期 (FastK)", 5, 25, 9)
    p2 = st.sidebar.slider("SlowK 磨平週期", 2, 10, 3)
    p3 = st.sidebar.slider("SlowD 磨平週期", 2, 10, 3)
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, 25)

# ==================== 4. 交易策略回測邏輯 ====================
def run_backtest(df, strategy, param1, param2, param3, sl_points):
    total_bars = len(df)
    rec = AssignmentRecord(total_bars)
    if total_bars < 2: return rec
    
    close = df['close'].values
    open_p = df['open'].values
    stop_loss_line = 0
    
    if "(一)" in strategy:
        ma_long = talib.SMA(df['close'], timeperiod=param1)
        ma_short = talib.SMA(df['close'], timeperiod=param2)
        for n in range(1, total_bars - 1):
            if np.isnan(ma_long[n-1]) or np.isnan(ma_short[n-1]): continue
            if rec.OpenInterestQty == 0:
                if ma_short[n-1] <= ma_long[n-1] and ma_short[n] > ma_long[n]:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif ma_short[n-1] >= ma_long[n-1] and ma_short[n] < ma_long[n]:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (ma_short[n-1] >= ma_long[n-1] and ma_short[n] < ma_long[n]) or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (ma_short[n-1] <= ma_long[n-1] and ma_short[n] > ma_long[n]) or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(二)" in strategy:
        rsi = talib.RSI(df['close'], timeperiod=param1)
        for n in range(1, total_bars - 1):
            if np.isnan(rsi[n-1]): continue
            if rec.OpenInterestQty == 0:
                if rsi[n-1] <= param2 and rsi[n] > param2:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
            elif rec.OpenInterestQty > 0:
                if (rsi[n-1] >= param2 and rsi[n] < param2) or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points

    elif "(三)" in strategy:
        rsi = talib.RSI(df['close'], timeperiod=param1)
        for n in range(1, total_bars - 1):
            if np.isnan(rsi[n-1]): continue
            if rec.OpenInterestQty == 0:
                if rsi[n-1] <= param2 and rsi[n] > param2:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif rsi[n-1] >= param3 and rsi[n] < param3:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if rsi[n] > param3 or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if rsi[n] < param2 or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(四)" in strategy:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=param1, nbdevup=param2, nbdevdn=param2)
        for n in range(1, total_bars - 1):
            if np.isnan(upper[n-1]): continue
            if rec.OpenInterestQty == 0:
                if close[n-1] <= lower[n-1] and close[n] > lower[n]:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif close[n-1] >= upper[n-1] and close[n] < upper[n]:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if close[n] > upper[n] or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if close[n] < lower[n] or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(五)" in strategy:
        _, _, macdhist = talib.MACD(df['close'], fastperiod=param1, slowperiod=param2, signalperiod=param3)
        for n in range(1, total_bars - 1):
            if np.isnan(macdhist[n-1]): continue
            hist_prev, hist_curr = macdhist[n-1], macdhist[n]
            if rec.OpenInterestQty == 0:
                if hist_prev <= 0 and hist_curr > 0:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif hist_prev >= 0 and hist_curr < 0:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (hist_prev >= 0 and hist_curr < 0) or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (hist_prev <= 0 and hist_curr > 0) or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(六)" in strategy:
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=param1, slowk_period=param2, slowk_matype=0, slowd_period=param3, slowd_matype=0)
        j_val = 3 * slowk - 2 * slowd
        for n in range(1, total_bars - 1):
            if np.isnan(slowk[n-1]): continue
            k_prev, k_curr = slowk[n-1], slowk[n]
            d_prev, d_curr = slowd[n-1], slowd[n]
            if rec.OpenInterestQty == 0:
                if k_prev <= d_prev and k_curr > d_curr and k_curr < 30:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif k_prev >= d_prev and k_curr < d_curr and k_curr > 70:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (k_prev >= d_prev and k_curr < d_curr) or j_val[n] > 100 or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (k_prev <= d_prev and k_curr > d_curr) or j_val[n] < 0 or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points
                    
    rec.FillRemainingEquity()
    return rec

# 執行回測
current_res = run_backtest(df_hourly, strategy_choice, p1, p2, p3, stop_loss)

# ==================== 5. 數據呈現儀表板 ====================
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 總淨利 (TWD)", f"${current_res.TotalProfit:,.0f}")
col2.metric("📈 交易勝率", f"{current_res.GetWinRate()*100:.2f}%")
col3.metric("📉 最大回撤 (MDD)", f"${current_res.MDD:,.0f}")
risk_reward = current_res.TotalProfit / current_res.MDD if current_res.MDD > 0 else 0
col4.metric("⚖️ 風險報酬比", f"{risk_reward:.2f}")

# 繪製【時空完全對齊】的真實圖表
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_hourly['time'], current_res.EquityHistory, label="累積盈虧歷史走勢", color="indigo", linewidth=2)
ax.set_title(f"{strategy_choice} - 累積資產權益曲線", fontsize=12)

ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.xticks(rotation=15)
st.pyplot(fig)

# ==================== 6. 參數最佳化功能 ====================
st.markdown("---")
st.header("🎯 機器智能參數最佳化功能")
if st.button("🚀 開始自動尋找最佳化參數組合"):
    with st.spinner("系統正在全速進行網格參數計算中..."):
        best_score = -999999
        best_p1, best_p2, best_p3 = p1, p2, p3
        
        if "(一)" in strategy_choice:
            p1_range, p2_range, p3_range = [20, 40, 60], [5, 10, 15], [0]
        elif "(二)" in strategy_choice:
            p1_range, p2_range, p3_range = [10, 14, 20], [50, 60, 70], [0]
        elif "(三)" in strategy_choice:
            p1_range, p2_range, p3_range = [10, 14, 20], [25, 30, 35], [65, 70, 75]
        elif "(四)" in strategy_choice:
            p1_range, p2_range, p3_range = [10, 20, 30], [1, 2, 3], [0]
        elif "(五)" in strategy_choice:
            p1_range, p2_range, p3_range = [6, 12, 18], [22, 26, 32], [7, 9, 12]
        else:
            p1_range, p2_range, p3_range = [5, 9, 14], [2, 3, 5], [2, 3, 5]
            
        for test_p1 in p1_range:
            for test_p2 in p2_range:
                for test_p3 in p3_range:
                    res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, test_p3, stop_loss)
                    score = res.TotalProfit / (res.MDD + 1)
                    if score > best_score:
                        best_score = score
                        best_p1, best_p2, best_p3 = test_p1, test_p2, test_p3
                        
        st.success(f"🎉 最佳化完成！")
        if "(五)" in strategy_choice or "(六)" in strategy_choice or "(三)" in strategy_choice:
            st.info(f"👉 建議最佳參數組合：參數1 = {best_p1}，參數2 = {best_p2}，參數3 = {best_p3}")
        else:
            st.info(f"👉 建議最佳參數組合：參數1 = {best_p1}，參數2 = {best_p2}")

# ==================== 7. AI 自動績效評估報告 ====================
st.markdown("---")
st.header("🤖 生成式 AI 策略績效自動評估報告")

def generate_ai_report(strat, profit, win_rate, mdd):
    status = "優異" if profit > 0 and win_rate > 0.30 else "需調整"
    suggestion = (
        "該量化策略成功捕捉了台積電的波段大趨勢。在小時線級別上展現出穩定的獲利因子。"
        if status == "優異" else 
        "目前參數在此級別暴露出過多假突破，導致利潤被侵蝕。建議在側邊欄適度拉大「移動止損點數」空間，給利潤更多奔跑的波動範圍。"
    )
    return f"""
    #### 【Generative AI 策略診斷書】
    * **評估對象策略**：{strat}
    * **目前策略綜合體質**：評級【**{status}**】
    * **量化數據診斷**：本策略回測最終創造利潤為 **${profit:,.0f} TWD**，交易勝率為 **{win_rate*100:.2f}%**，承受的最大潛在資金回落(MDD)為 **${mdd:,.0f} TWD**。
    * **AI 深度優化修補建議**：{suggestion}
    """
st.markdown(generate_ai_report(strategy_choice, current_res.TotalProfit, current_res.GetWinRate(), current_res.MDD))