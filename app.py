import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Nigeria Food Price Intelligence",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1565C0;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .signal-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1565C0;
        border-left: 5px solid #1565C0;
        padding-left: 0.7rem;
        margin-bottom: 0.3rem;
    }
    .signal-subtitle {
        font-size: 1.0rem;
        color: #37474F;
        font-style: italic;
        margin-bottom: 1rem;
    }
    .emergence-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1565C0;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #1A237E;
    }
    .sell-box {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #1B5E20;
    }
    .metric-card {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .stMetric label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
MONTH_COLORS = {
    'Oct 2025': '#2196F3',
    'Nov 2025': '#FF9800',
    'Dec 2025': '#F44336'
}
VLINE_STYLE = dict(linestyle='--', alpha=0.45, linewidth=1.2)
sns.set_theme(style='whitegrid', font_scale=1.05)

DATA_DIR = 'data'

# ============================================================
# CSV PARSER — same as Colab version
# ============================================================
@st.cache_data
def parse_netlogo_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read().replace('\r\n', '\n').replace('\r', '\n')
    lines = raw.split('\n')

    pen_names = []
    for i, line in enumerate(lines):
        if '"pen name"' in line:
            j = i + 1
            while j < len(lines) and lines[j].strip():
                name = lines[j].strip().split(',')[0].strip().strip('"')
                if name:
                    pen_names.append(name)
                j += 1
            break

    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('"x","y"') or line.startswith('x,y,'):
            data_start = i + 1
            break

    if data_start is None:
        return pd.DataFrame()

    data_lines = '\n'.join([l for l in lines[data_start:] if l.strip()])
    raw_df = pd.read_csv(io.StringIO(data_lines), header=None)

    result = pd.DataFrame()
    result['tick'] = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce').astype(int)
    for i, pen in enumerate(pen_names):
        y_col = i * 4 + 1
        result[pen] = pd.to_numeric(raw_df.iloc[:, y_col], errors='coerce')

    result = result.dropna(subset=['tick']).reset_index(drop=True)
    result['month'] = pd.cut(
        result['tick'],
        bins=[-1, 29, 59, 200],
        labels=['Oct 2025', 'Nov 2025', 'Dec 2025']
    )
    return result

# ============================================================
# LOAD ALL DATA
# ============================================================
@st.cache_data
def load_all_data():
    files = {
        'p1': 'plot1_qvalue_convergence.csv',
        'p2': 'plot2_trader_flow.csv',
        'p3': 'plot3_arbitrage_margin.csv',
        'p4': 'plot4_stress_cascade.csv',
        'p5': 'plot5_tomato_prices.csv',
        'p6': 'plot6_buyer_retention.csv',
        'p7': 'plot7_ctde_learning.csv',
        'p8': 'plot8_trader_status.csv',
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            data[key] = parse_netlogo_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data

# ============================================================
# HELPER: month band shading
# ============================================================
def add_month_bands(ax):
    ax.axvline(x=30, color='gray',  **VLINE_STYLE, label='Nov start')
    ax.axvline(x=60, color='black', **VLINE_STYLE, label='Dec start')
    ax.axvspan(0,  30, alpha=0.04, color='#2196F3')
    ax.axvspan(30, 60, alpha=0.04, color='#FF9800')
    ax.axvspan(60, 92, alpha=0.04, color='#F44336')

# ============================================================
# CHART FUNCTIONS — one per signal
# ============================================================

def chart_signal1(df, month_filter):
    if df.empty:
        return None
    d = df if month_filter == 'All' else df[df['month'] == month_filter]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(d['tick'], d['Q sell-local'],    color='#4CAF50', lw=2, label='Q sell-local')
    ax.plot(d['tick'], d['Q travel-to-NW'], color='#2196F3', lw=2, label='Q travel-to-NW')
    ax.plot(d['tick'], d['SW Advantage'],   color='#F44336', lw=1.3, label='SW Advantage', alpha=0.8)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Q-Value')
    ax.set_title('Q-Values Over Time (SW Traders)')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    monthly = df.groupby('month', observed=True)[['Q sell-local', 'Q travel-to-NW']].mean()
    x = np.arange(3); w = 0.35
    ax2.bar(x - w/2, monthly['Q sell-local'],    w, color='#4CAF50', label='Q sell-local',    edgecolor='white')
    ax2.bar(x + w/2, monthly['Q travel-to-NW'], w, color='#2196F3', label='Q travel-to-NW', edgecolor='white')
    ax2.set_xticks(x); ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
    ax2.set_title('Mean Q-Values by Month'); ax2.set_ylabel('Mean Q-Value')
    ax2.legend(fontsize=9); ax2.tick_params(axis='x', rotation=10)
    plt.tight_layout()
    return fig

def chart_signal2(df, month_filter):
    if df.empty:
        return None
    flow_pens = [c for c in df.columns if c.startswith('SW in')]
    flow_colors = ['#1565C0', '#388E3C', '#D32F2F', '#7B1FA2', '#F57C00']
    d = df if month_filter == 'All' else df[df['month'] == month_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for pen, col in zip(flow_pens, flow_colors):
        ax.plot(d['tick'], d[pen], color=col, lw=1.8, label=pen, alpha=0.85)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('SW Traders in Region')
    ax.set_title('SW Trader Destinations Each Tick')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    monthly_flow = df.groupby('month', observed=True)[flow_pens].mean()
    bottom = np.zeros(3)
    for pen, col in zip(flow_pens, flow_colors):
        vals = monthly_flow[pen].values
        ax2.bar(range(3), vals, bottom=bottom, color=col,
                label=pen, edgecolor='white', linewidth=0.5)
        bottom += vals
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
    ax2.set_title('Mean SW Traders Away — Stacked by Destination')
    ax2.set_ylabel('Mean Traders Away from SW')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.tick_params(axis='x', rotation=10)
    plt.tight_layout()
    return fig

def chart_signal3(df, month_filter):
    if df.empty:
        return None
    gap_cols = [c for c in df.columns if c not in ['tick', 'month']]
    d = df if month_filter == 'All' else df[df['month'] == month_filter]
    transport = [8770.49, 8917.70, 9080.95]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    colors = ['#2196F3', '#F44336']
    for i, col in enumerate(gap_cols):
        ax.plot(d['tick'], d[col], color=colors[i % len(colors)], lw=2, label=col)
    ax.axhline(y=0, color='black', lw=0.8, alpha=0.5)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Price Gap (NGN)')
    ax.set_title('SW vs NW/SE Tomato Price Gap Over Time')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if gap_cols:
        monthly_gap = df.groupby('month', observed=True)[gap_cols[0]].mean()
        ax2.bar(range(3), monthly_gap.values,
                color=['#2196F3', '#FF9800', '#F44336'],
                edgecolor='white', linewidth=1.5, width=0.5,
                label='Mean SW-NW Gap')
    ax2.plot(range(3), transport, 'ko--', lw=2, ms=8, label='Bus fare threshold (NGN)')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
    ax2.set_title('Mean Price Gap vs Bus Fare Threshold')
    ax2.set_ylabel('NGN'); ax2.legend(fontsize=9)
    ax2.tick_params(axis='x', rotation=10)
    plt.tight_layout()
    return fig

def chart_signal4(df, month_filter):
    if df.empty:
        return None
    rew_cols = [c for c in df.columns if c not in ['tick', 'month', 'stress_score']]
    rew_colors = {'SW winning': '#4CAF50', 'SW losing': '#F44336',
                  'SW transporters': '#FF9800', 'SW buyers': '#2196F3'}
    d = df if month_filter == 'All' else df[df['month'] == month_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for pen in rew_cols:
        col = rew_colors.get(pen, '#607D8B')
        ax.plot(d['tick'], d[pen], color=col, lw=2, label=pen)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative Reward (NGN)')
    ax.set_title('SW Agent Cumulative Rewards Over Time')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    win_cols = [c for c in rew_cols if 'win' in c.lower()]
    los_cols = [c for c in rew_cols if 'los' in c.lower()]
    if win_cols and los_cols:
        df = df.copy()
        df['stress_score'] = (
            df[los_cols[0]].abs() /
            (df[win_cols[0]].abs() + df[los_cols[0]].abs() + 1)
        )
        monthly_stress = df.groupby('month', observed=True)['stress_score'].mean()
        bars = ax2.bar(range(3), monthly_stress.values,
                       color=['#2196F3', '#FF9800', '#F44336'],
                       edgecolor='white', linewidth=1.5, width=0.5)
        ax2.axhline(y=0.5, color='red', linestyle='--', lw=1.5, label='Equal stress (0.5)')
        ax2.set_title('Supply Chain Stress Score by Month')
        ax2.set_xlabel('Month'); ax2.set_ylabel('Stress Score (0–1)')
        ax2.set_ylim(0, 1.05); ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
        ax2.tick_params(axis='x', rotation=10); ax2.legend(fontsize=9)
        for bar, val in zip(bars, monthly_stress.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.015,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def chart_signal5(df, month_filter):
    if df.empty:
        return None
    price_map = {
        'SW PMS':           ('#F44336', 'SW Tomatoes'),
        'NE PMS':           ('#2196F3', 'NW Tomatoes'),
        'SW Bus intercity': ('#FF9800', 'NE Tomatoes'),
        'NW Bus intercity': ('#4CAF50', 'NC Tomatoes'),
    }
    d = df if month_filter == 'All' else df[df['month'] == month_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for pen, (col, label) in price_map.items():
        if pen in d.columns:
            ax.plot(d['tick'], d[pen], color=col, lw=2, label=label)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Tomato Price (NGN)')
    ax.set_title('Regional Tomato Prices Over Time')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    pens = [p for p in price_map if p in df.columns]
    monthly_prices = df.groupby('month', observed=True)[pens].mean()
    x = np.arange(3); w = 0.2
    for idx, pen in enumerate(pens):
        label = price_map[pen][1]
        col   = price_map[pen][0]
        offset = (idx - 1.5) * w
        ax2.bar(x + offset, monthly_prices[pen].values, w,
                color=col, label=label, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
    ax2.set_title('Mean Tomato Price by Region and Month')
    ax2.set_ylabel('Mean Price (NGN)')
    ax2.legend(fontsize=9); ax2.tick_params(axis='x', rotation=10)
    plt.tight_layout()
    return fig

def chart_signal6(df, month_filter):
    if df.empty:
        return None
    d = df if month_filter == 'All' else df[df['month'] == month_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(d['tick'], d['SW buy-local rate'],
            color='#4CAF50', lw=2, label='Buy-local rate')
    ax.plot(d['tick'], d['SW request-adjacent rate'],
            color='#F44336', lw=2, label='Request-adjacent rate')
    ax.axhline(y=0.5, color='gray', linestyle='--', lw=1.2, alpha=0.6, label='Equal split (0.5)')
    ax.axhline(y=0.6, color='green', linestyle=':', lw=1.2, alpha=0.7, label='Price floor threshold (0.6)')
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Action Rate (0–1)')
    ax.set_title('SW Buyer Action Rates Over Time')
    ax.legend(fontsize=9); ax.set_ylim(-0.05, 1.1)

    ax2 = axes[1]
    monthly_ret = df.groupby('month', observed=True)['SW buy-local rate'].mean()
    bars = ax2.bar(range(3), monthly_ret.values,
                   color=['#2196F3', '#FF9800', '#F44336'],
                   edgecolor='white', linewidth=1.5, width=0.5)
    ax2.axhline(y=0.6, color='green', linestyle='--', lw=1.5, label='Price floor threshold (0.6)')
    ax2.set_title('Mean Buy-Local Rate by Month\n(Above 0.6 = Price Floor Forming)')
    ax2.set_ylabel('Mean Buy-Local Rate'); ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
    ax2.tick_params(axis='x', rotation=10); ax2.legend(fontsize=9)
    for bar, val in zip(bars, monthly_ret.values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.015,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def chart_signal7(df, month_filter):
    if df.empty:
        return None
    ctde_map = {
        'SW traders (high CTDE prior)': ('#FF9800', 'SW (High CTDE Prior)'),
        'NE traders (low CTDE prior)':  ('#2196F3', 'NE (Low CTDE Prior)'),
        'NC traders':                   ('#4CAF50', 'NC (Mid Prior)'),
    }
    d = df if month_filter == 'All' else df[df['month'] == month_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for pen, (col, label) in ctde_map.items():
        if pen in d.columns:
            ax.plot(d['tick'], d[pen], color=col, lw=2, label=label)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative Reward (NGN)')
    ax.set_title('Cumulative Reward by CTDE Prior Level')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    sw_col = 'SW traders (high CTDE prior)'
    ne_col = 'NE traders (low CTDE prior)'
    if sw_col in df.columns and ne_col in df.columns:
        df = df.copy()
        df['ctde_gap'] = df[sw_col] - df[ne_col]
        monthly_gap = df.groupby('month', observed=True)['ctde_gap'].mean()
        bars = ax2.bar(range(3), monthly_gap.values,
                       color=['#2196F3', '#FF9800', '#F44336'],
                       edgecolor='white', linewidth=1.5, width=0.5)
        ax2.set_title('CTDE Learning Gap by Month\n(SW Reward minus NE Reward)')
        ax2.set_ylabel('Reward Gap (NGN)'); ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
        ax2.tick_params(axis='x', rotation=10)
        ax2.axhline(y=0, color='black', lw=0.8)
        for bar, val in zip(bars, monthly_gap.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(abs(monthly_gap.values)) * 0.02,
                     f'{val:,.0f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig

def chart_signal8(df, month_filter):
    if df.empty:
        return None
    status_map = {
        'SW winners count': ('#4CAF50', 'SW Winners'),
        'SW losers count':  ('#F44336', 'SW Losers'),
        'SW traders away':  ('#2196F3', 'SW Away'),
        'NW traders away':  ('#00BCD4', 'NW Away'),
        'NE traders away':  ('#FF9800', 'NE Away'),
    }
    d = df if month_filter == 'All' else df[df['month'] == month_filter]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for pen, (col, label) in status_map.items():
        if pen in d.columns:
            ax.plot(d['tick'], d[pen], color=col, lw=2, label=label, alpha=0.85)
    add_month_bands(ax)
    ax.set_xlabel('Tick'); ax.set_ylabel('Number of Traders')
    ax.set_title('SW Trader Status Over Time')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    win_col  = [c for c in status_map if 'win' in c.lower() and c in df.columns]
    los_col  = [c for c in status_map if 'los' in c.lower() and c in df.columns]
    away_col = [c for c in status_map if 'away' in c.lower() and 'sw' in c.lower() and c in df.columns]

    if win_col and los_col:
        df = df.copy()
        df['wl_ratio'] = df[win_col[0]] / (df[los_col[0]] + 0.001)
        monthly_wl   = df.groupby('month', observed=True)['wl_ratio'].mean()
        monthly_away = df.groupby('month', observed=True)[away_col[0]].mean() if away_col else None

        bars = ax2.bar(range(3), monthly_wl.values,
                       color=['#2196F3', '#FF9800', '#F44336'],
                       edgecolor='white', linewidth=1.5, width=0.5,
                       label='Winner/Loser Ratio', zorder=3)
        ax2.axhline(y=1.0, color='red', linestyle='--', lw=1.5, label='Equal ratio (1.0)')
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Oct 2025', 'Nov 2025', 'Dec 2025'])
        ax2.set_title('Winner/Loser Ratio & Traders Away\n(Ratio < 1 = Stress Signal)')
        ax2.set_ylabel('Winner / Loser Ratio')
        ax2.tick_params(axis='x', rotation=10); ax2.legend(fontsize=9, loc='upper left')

        if monthly_away is not None:
            ax2b = ax2.twinx()
            ax2b.plot(range(3), monthly_away.values, 'bs--',
                      lw=2, ms=9, label='SW Traders Away (mean)')
            ax2b.set_ylabel('Mean SW Traders Away', color='#2196F3')
            ax2b.tick_params(axis='y', labelcolor='#2196F3')
            ax2b.legend(fontsize=9, loc='upper right')

        for bar, val in zip(bars, monthly_wl.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig

def chart_dashboard(data):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#FAFAFA')

    # Panel 1 — Q convergence
    ax1 = fig.add_subplot(2, 3, 1)
    p1 = data['p1']
    if not p1.empty:
        ax1.plot(p1['tick'], p1['Q sell-local'],    color='#4CAF50', lw=2, label='Q sell-local')
        ax1.plot(p1['tick'], p1['Q travel-to-NW'], color='#2196F3', lw=2, label='Q travel-to-NW')
        ax1.axvline(30, color='gray',  **VLINE_STYLE)
        ax1.axvline(60, color='black', **VLINE_STYLE)
    ax1.set_title('Signal 1: Q-Value Convergence', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Tick'); ax1.legend(fontsize=7)

    # Panel 2 — trader flow
    ax2 = fig.add_subplot(2, 3, 2)
    p2 = data['p2']
    if not p2.empty:
        flow_pens = [c for c in p2.columns if c.startswith('SW in')]
        p2 = p2.copy()
        p2['total'] = p2[flow_pens].sum(axis=1)
        m2 = p2.groupby('month', observed=True)['total'].mean()
        ax2.bar(range(3), m2.values,
                color=['#2196F3', '#FF9800', '#F44336'], edgecolor='white')
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Oct', 'Nov', 'Dec'])
    ax2.set_title('Signal 2: Arbitrage Flow Collapse', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Mean SW Traders Away')

    # Panel 3 — arbitrage margin
    ax3 = fig.add_subplot(2, 3, 3)
    p3 = data['p3']
    if not p3.empty:
        gap_cols = [c for c in p3.columns if c not in ['tick', 'month']]
        g_colors = ['#2196F3', '#F44336']
        for i, col in enumerate(gap_cols):
            ax3.plot(p3['tick'], p3[col], color=g_colors[i % 2], lw=2, label=col)
        ax3.axhline(0, color='black', lw=0.8, alpha=0.5)
        ax3.axvline(30, color='gray',  **VLINE_STYLE)
        ax3.axvline(60, color='black', **VLINE_STYLE)
    ax3.set_title('Signal 3: Arbitrage Margin NGN', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Tick'); ax3.legend(fontsize=7)

    # Panel 4 — stress score
    ax4 = fig.add_subplot(2, 3, 4)
    p4 = data['p4']
    if not p4.empty:
        win_c = [c for c in p4.columns if 'win' in c.lower()]
        los_c = [c for c in p4.columns if 'los' in c.lower()]
        if win_c and los_c:
            p4 = p4.copy()
            p4['stress_score'] = (
                p4[los_c[0]].abs() /
                (p4[win_c[0]].abs() + p4[los_c[0]].abs() + 1)
            )
            m4 = p4.groupby('month', observed=True)['stress_score'].mean()
            bars4 = ax4.bar(range(3), m4.values,
                            color=['#2196F3', '#FF9800', '#F44336'],
                            edgecolor='white', width=0.5)
            ax4.axhline(0.5, color='red', linestyle='--', lw=1.5)
            ax4.set_xticks(range(3))
            ax4.set_xticklabels(['Oct', 'Nov', 'Dec'])
            ax4.set_ylim(0, 1.0)
            for bar, val in zip(bars4, m4.values):
                ax4.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.02, f'{val:.3f}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_title('Signal 4: Stress Score', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Stress Score (0–1)')

    # Panel 5 — tomato prices
    ax5 = fig.add_subplot(2, 3, 5)
    p5 = data['p5']
    price_cfg = [
        ('SW PMS',           '#F44336', 'SW Tomatoes'),
        ('NE PMS',           '#2196F3', 'NW Tomatoes'),
        ('SW Bus intercity', '#FF9800', 'NE Tomatoes'),
        ('NW Bus intercity', '#4CAF50', 'NC Tomatoes'),
    ]
    if not p5.empty:
        for pen, col, label in price_cfg:
            if pen in p5.columns:
                ax5.plot(p5['tick'], p5[pen], color=col, lw=2, label=label)
        ax5.axvline(30, color='gray',  **VLINE_STYLE)
        ax5.axvline(60, color='black', **VLINE_STYLE)
    ax5.set_title('Signal 5: Regional Tomato Prices', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Tick'); ax5.set_ylabel('NGN')
    ax5.legend(fontsize=7)

    # Panel 6 — CTDE gap
    ax6 = fig.add_subplot(2, 3, 6)
    p7 = data['p7']
    if not p7.empty:
        sw_c = 'SW traders (high CTDE prior)'
        ne_c = 'NE traders (low CTDE prior)'
        if sw_c in p7.columns and ne_c in p7.columns:
            p7 = p7.copy()
            p7['ctde_gap'] = p7[sw_c] - p7[ne_c]
            m7 = p7.groupby('month', observed=True)['ctde_gap'].mean()
            bars6 = ax6.bar(range(3), m7.values,
                            color=['#2196F3', '#FF9800', '#F44336'],
                            edgecolor='white', width=0.5)
            ax6.axhline(0, color='black', lw=0.8)
            ax6.set_xticks(range(3))
            ax6.set_xticklabels(['Oct', 'Nov', 'Dec'])

    ax6.set_title('Signal 7: CTDE Learning Advantage', fontweight='bold', fontsize=11)
    ax6.set_ylabel('SW minus NE Reward (NGN)')

    oct_p = mpatches.Patch(color='#2196F3', alpha=0.6, label='October 2025')
    nov_p = mpatches.Patch(color='#FF9800', alpha=0.6, label='November 2025')
    dec_p = mpatches.Patch(color='#F44336', alpha=0.6, label='December 2025')
    fig.legend(handles=[oct_p, nov_p, dec_p],
               loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(
        'Nigeria Food Price Intelligence — Emergence Signals Dashboard\n'
        'SW Tomato Market · October–December 2025 · NBS Data + MARL Simulation',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    return fig

# ============================================================
# SIGNAL METADATA
# ============================================================
SIGNALS = {
    'Signal 1 — Q-Value Convergence': {
        'key': 'p1',
        'chart_fn': chart_signal1,
        'subtitle': 'NW Arbitrage Corridor Discovery',
        'emergence': (
            'SW traders were not told that selling locally is more profitable than travelling. '
            'They discovered it through Q-learning. Q sell-local pulling ahead of Q travel-to-NW '
            'in December is the emergence event — occurring without any rule directing agent preference. '
            'In December, the arbitrage margin from NW becomes negative: ₦1,189 minus ₦790 minus ₦9,081 fare = '
            '−₦8,682. Agents received negative rewards for travel and rapidly updated Q travel-to-NW downward.'
        ),
        'sell': (
            'When Q sell-local consistently exceeds Q travel-to-NW, the SW tomato market has stabilised. '
            'Leading indicator of price floor formation — detectable 2–4 weeks before aggregate NBS statistics show it.'
        ),
        'metrics': [
            ('Q sell-local (Dec mean)', '~₦2.5M'),
            ('Q travel-to-NW (Dec mean)', '~₦1.0M'),
            ('SW Advantage gain', '+114% SW vs NE'),
        ]
    },
    'Signal 2 — Arbitrage Flow Map': {
        'key': 'p2',
        'chart_fn': chart_signal2,
        'subtitle': 'December Arbitrage Collapse',
        'emergence': (
            'The destination shift from NW in October to NE in December was not programmed. '
            'It emerged from Q-learning adapting to December\'s price structure. '
            'NE Reg3 β recovered from −0.169 in November to +0.138 in December — agents detected '
            'this recovery through Q-table updates and shifted travel behaviour accordingly.'
        ),
        'sell': (
            'The stacked destination map is a real-time supply chain routing signal. '
            'Which region is supplying SW tomatoes this quarter? '
            'Quarterly supply chain routing map showing which regions are supplying SW tomatoes. Updated each quarter as new NBS data is released.'
        ),
        'metrics': [
            ('Oct dominant destination', 'North West (Reg3 β=+0.782)'),
            ('Dec dominant destination', 'North East (β recovery)'),
            ('Mean traders away (Dec)', '~0.73'),
        ]
    },
    'Signal 3 — Arbitrage Margin': {
        'key': 'p3',
        'chart_fn': chart_signal3,
        'subtitle': 'Transport Cost Threshold Analysis',
        'emergence': (
            'In all three months, the mean SW-NW price gap is far below the bus fare threshold. '
            'This means the market never generated a sustained rational arbitrage opportunity — '
            'and agents learned this correctly. The two plots validate each other: '
            'price data explains why Q travel-to-NW declined relative to Q sell-local (Signal 1).'
        ),
        'sell': (
            'The Arbitrage Viability Index — quantifying when inter-regional tomato trade is '
            'economically rational. Published quarterly alongside NBS transport cost data as the Arbitrage Viability Index.'
        ),
        'metrics': [
            ('Oct gap vs fare', '₦474 vs ₦8,770 (5.4% of fare)'),
            ('Nov gap vs fare', 'Spike-inflated vs ₦8,918'),
            ('Dec gap vs fare', '₦399 vs ₦9,081 (4.4% of fare)'),
        ]
    },
    'Signal 4 — Stress Cascade': {
        'key': 'p4',
        'chart_fn': chart_signal4,
        'subtitle': 'WoLF-PHC Winner/Loser Bifurcation',
        'emergence': (
            'The stress score peaking in November (0.453), not December, is the emergence insight. '
            'November is when Reg2 β flips to cost-push (−0.086) and buyer uncertainty peaks. '
            'The combined pressures create maximum agent stress before the December price data '
            'is even loaded — a predictive lead signal not visible in NBS aggregates.'
        ),
        'sell': (
            'Supply Chain Stress Score — a stress score above 0.5 is a lead indicator of food supply disruption, '
            'detectable in agent behaviour 2–4 weeks before aggregate price statistics show the stress.'
        ),
        'metrics': [
            ('Oct stress score', '0.405'),
            ('Nov stress score', '0.453 ← Peak'),
            ('Dec stress score', '0.405'),
        ]
    },
    'Signal 5 — Regional Tomato Prices': {
        'key': 'p5',
        'chart_fn': chart_signal5,
        'subtitle': 'Arbitrage Opportunity Map Across Regions',
        'emergence': (
            'All four regional prices converging downward in December is the emergence signal — '
            'the market-wide supply glut transmitted through agent behaviour across the simulation. '
            'NE falling fastest (₦1,111 to ₦977) explains why SW traders shifted to NE as a '
            'destination in December (Signal 2), even though the November Reg3 NE β was negative.'
        ),
        'sell': (
            'Quarterly Regional Price Intelligence Map — showing where tomatoes are cheapest and how regional prices '
            'co-move. Embedded in consumer price dashboards as a price comparison layer.'
        ),
        'metrics': [
            ('Cheapest region', 'NW: ₦790 (Dec)'),
            ('Most expensive', 'SW: ₦1,189 (Dec)'),
            ('Most volatile', 'NE: −11.7% Oct→Dec'),
        ]
    },
    'Signal 6 — JAL-AM Buyer Retention': {
        'key': 'p6',
        'chart_fn': chart_signal6,
        'subtitle': 'Price Floor Formation Signal',
        'emergence': (
            'A buy-local rate above 0.6 means SW buyers collectively decided that local prices '
            'are competitive enough that alternatives are not worth the travel cost. '
            'The SW tomato market established stable buyer retention from tick 1 and maintained '
            'it through the December transport shock — not programmed, emerged from buyer Q-learning.'
        ),
        'sell': (
            'Price Floor Formation Signal — SW tomato market maintained 88–94% buyer retention Oct–Dec 2025, '
            'confirming a stable demand floor at ₦1,189. Actionable for FMCG procurement teams.'
        ),
        'metrics': [
            ('Oct buy-local rate', '93.8%'),
            ('Nov buy-local rate', '92.7%'),
            ('Dec buy-local rate', '88.0%'),
        ]
    },
    'Signal 7 — CTDE Learning Advantage': {
        'key': 'p7',
        'chart_fn': chart_signal7,
        'subtitle': 'Market Intelligence Index',
        'emergence': (
            'The growing SW-NE gap was not programmed — it emerged from agents with better-informed '
            'starting Q-values (from Reg1 regression priors) making compoundingly better decisions. '
            'October gap is slightly negative (NE briefly ahead) showing CTDE priors only '
            'fully materialise once agents shift from exploring to exploiting in November and December.'
        ),
        'sell': (
            'Market Intelligence Index — markets with stronger NBS price data coverage demonstrate measurably '
            'faster price discovery. SW agents outperformed NE by 114% in December 2025. '
            'Updated quarterly as new NBS data is released.'
        ),
        'metrics': [
            ('Oct CTDE gap', '−₦26,840 (NE briefly ahead)'),
            ('Nov CTDE gap', '+₦1.4M (SW leads)'),
            ('Dec CTDE gap', '+₦25.2M (114% advantage)'),
        ]
    },
    'Signal 8 — Trader Status Count': {
        'key': 'p8',
        'chart_fn': chart_signal8,
        'subtitle': 'Winner-Loser-Away Three-Way Stress Signal',
        'emergence': (
            'The winner-loser ratio falling from 1.63 to 0.35 across three months is the most '
            'powerful single emergence signal: a full market stress inversion. '
            'Three simultaneous December pressures drove it: prices fell 13%, fares rose 3.5%, '
            'Reg2 β flipped negative. The rising traders-away count despite negative arbitrage '
            'margins is the ε=0.20 exploration effect — maximum uncertainty drives maximum search.'
        ),
        'sell': (
            'Three-way stress signal — when winner ratio falls below 1.0 + losers rising + exploration '
            'rising simultaneously, a food supply chain stress event is underway before NBS '
            'aggregate statistics capture it. Early warning product.'
        ),
        'metrics': [
            ('Oct winner/loser ratio', '1.63 — profitable market'),
            ('Nov winner/loser ratio', '1.37 — declining'),
            ('Dec winner/loser ratio', '0.35 — stress inversion'),
        ]
    },
}

# ============================================================
# MAIN APP
# ============================================================

def main():
    # ── Header ───────────────────────────────────────────────
    st.markdown('<div class="main-title">🇳🇬 Nigeria Food Price Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">SW Tomato Market · October–December 2025 · '
        'NBS Data + Multi-Agent Reinforcement Learning Simulation</div>',
        unsafe_allow_html=True
    )

    # ── Load data ─────────────────────────────────────────────
    data = load_all_data()

    # Check if data folder exists
    missing = [k for k, v in data.items() if v.empty]
    if missing:
        st.error(
            f'⚠️ Missing data files: {missing}. '
            f'Please ensure all 8 CSV files are in the `{DATA_DIR}/` folder.'
        )

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Flag_of_Nigeria.svg/200px-Flag_of_Nigeria.svg.png', width=80)
        st.markdown('### 🔍 Controls')

        view = st.radio(
            'View mode',
            ['Individual Signals', 'Full Dashboard'],
            index=0
        )

        if view == 'Individual Signals':
            selected_signal = st.selectbox(
                'Select signal',
                list(SIGNALS.keys()),
                index=0
            )
            month_filter = st.selectbox(
                'Month filter',
                ['All', 'Oct 2025', 'Nov 2025', 'Dec 2025'],
                index=0
            )
    

        st.markdown('---')
        st.markdown('### 📊 About')
        st.markdown(
            'This dashboard presents synthetic behavioural intelligence '
            'derived from a MARL simulation of Nigerian food, energy and '
            'transport markets using real NBS data.'
        )
        st.markdown('**Algorithms:** Q-Learning · JAL-AM · WoLF-PHC · CTDE')
        st.markdown('**Agents:** 366 (Traders · Buyers · Transporters · Shock)')
        st.markdown('**Data:** NBS CPI Oct–Dec 2025 · 945 rows · 6 regions')

    # ── Full Dashboard view ───────────────────────────────────
    if view == 'Full Dashboard':
        st.markdown('## 📋 Full Emergence Signals Dashboard')
        st.markdown(
            'All six key signals in one view. Vertical dashed lines mark '
            'the October→November (tick 30) and November→December (tick 60) transitions.'
        )
        with st.spinner('Generating dashboard...'):
            fig = chart_dashboard(data)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Download button
        buf = io.BytesIO()
        fig2 = chart_dashboard(data)
        fig2.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            '⬇️ Download Dashboard PNG',
            data=buf,
            file_name='nigeria_marl_dashboard.png',
            mime='image/png'
        )
        plt.close()
        return

    # ── Individual Signal view ────────────────────────────────
    sig_info = SIGNALS[selected_signal]

    # Signal header
    st.markdown(f'<div class="signal-title">{selected_signal}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="signal-subtitle">{sig_info["subtitle"]}</div>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3 = st.columns(3)
    for col, (label, val) in zip([col1, col2, col3], sig_info['metrics']):
        with col:
            st.metric(label=label, value=val)

    st.markdown('---')

    # Chart
    df = data[sig_info['key']]
    with st.spinner('Generating chart...'):
        fig = sig_info['chart_fn'](df, month_filter)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

            # Download chart
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                f'⬇️ Download Chart PNG',
                data=buf,
                file_name=f'{selected_signal.lower().replace(" ", "_").replace("—","").strip()}.png',
                mime='image/png'
            )
            plt.close()

    st.markdown('---')

    # Emergence explanation
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('#### 🔬 Where Emergence Arises')
        st.markdown(
            f'<div class="emergence-box">{sig_info["emergence"]}</div>',
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown('#### 💼 Intelligence Product')
        st.markdown(
            f'<div class="sell-box">{sig_info["sell"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown('---')

    # Download CSV
    if not df.empty:
        csv_buf = df.to_csv(index=False)
        st.download_button(
            '⬇️ Download Signal Data CSV',
            data=csv_buf,
            file_name=f'signal_data_{sig_info["key"]}.csv',
            mime='text/csv'
        )

    # Navigation
    st.markdown('---')
    st.markdown('#### Navigate to another signal')
    sig_keys = list(SIGNALS.keys())
    cols = st.columns(4)
    for i, sig_name in enumerate(sig_keys):
        with cols[i % 4]:
            st.markdown(f'📊 **{sig_name.split("—")[0].strip()}**')

    st.markdown('---')
    st.markdown(
        '<small>Data: NBS CPI Oct–Dec 2025 · Simulation: NetLogo 6.4 · '
        'Algorithms: Albrecht, Christianos et al. MARL (MIT Press, 2024)</small>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
