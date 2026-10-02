import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import fitparse

KLEUREN_PALETTE = {
    'background': '#000000',
    'card_bg': '#1E222D',
    'hr': '#FF2A6D',          # Neon Coral
    'speed': '#00F5FF',       # Ice Cyan
    'bike': '#FFD700',        # Gold
    'muted': '#8A99AD',       # Slate Gray
    'grid': '#2A2A2A',
    'text': '#FFFFFF'
}

def read_fit_file(file_path: str) -> pd.DataFrame:
    """Leest een .fit bestand uit en geeft de record-data terug als Pandas DataFrame."""
    fitfile = fitparse.FitFile(file_path)
    records = [{data.name: data.value for data in record} for record in fitfile.get_messages('record')]
    return pd.DataFrame(records)

parse_fit = read_fit_file

def clean_fit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Schoont de FIT-dataframe op: zet snelheden om en converteert timestamps."""
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    if 'enhanced_speed' in df.columns and df['enhanced_speed'].notna().any():
        df['speed_kmh'] = df['enhanced_speed'] * 3.6
    elif 'speed' in df.columns:
        df['speed_kmh'] = df['speed'] * 3.6
    else:
        df['speed_kmh'] = 0.0

    for col in ['heart_rate', 'power', 'cadence', 'distance']:
        if col not in df.columns:
            df[col] = None

    return df

def calculate_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent de verstreken tijd in minuten vanaf de start van de activiteit."""
    df = df.copy()
    if 'timestamp' in df.columns and not df['timestamp'].empty:
        start_time = df['timestamp'].iloc[0]
        df['elapsed_minutes'] = (df['timestamp'] - start_time).dt.total_seconds() / 60.0
    else:
        df['elapsed_minutes'] = 0.0
    return df

def apply_triathlon_style():
    """Stelt de uniforme merkidentiteit in voor Matplotlib visuals."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'figure.facecolor': KLEUREN_PALETTE['background'],
        'axes.facecolor': KLEUREN_PALETTE['background'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'grid.color': KLEUREN_PALETTE['grid'],
        'grid.linestyle': ':',
        'grid.alpha': 0.6,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': KLEUREN_PALETTE['muted'],
        'ytick.color': KLEUREN_PALETTE['muted']
    })

def format_time_axis(ax, label="Tijd (minuten)", add_suffix=False):
    """Zorgt voor een uniforme x-as opmaak."""
    ax.set_xlabel(label, fontweight='bold', fontsize=10, color=KLEUREN_PALETTE['muted'], labelpad=8)
    if add_suffix:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}m"))
    else:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))

def plot_race_anatomy_template(df, fases):
    """Genereert de Race Opbouw visualisatie met fase-kleurvlakken."""
    apply_triathlon_style()
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    max_hr = df['heart_rate'].max() if df['heart_rate'].notna().any() else 180
    min_hr = df['heart_rate'].min() if df['heart_rate'].notna().any() else 100

    # 1. Fase-achtergronden en labels
    for fase in fases:
        ax1.axvline(fase['start'], color='#333745', linestyle=':', linewidth=1.0)
        midden = (fase['start'] + fase['eind']) / 2
        ax1.text(midden, max_hr + 5, fase['naam'], 
                 ha='center', va='bottom', fontweight='bold', color=KLEUREN_PALETTE['muted'], fontsize=9)
    
    if fases:
        ax1.axvline(fases[-1]['eind'], color='#333745', linestyle=':', linewidth=1.0)

    # 2. Hartslaglijn
    ax1.plot(df['elapsed_minutes'], df['heart_rate'], color=KLEUREN_PALETTE['hr'], linewidth=1.8, label='Hartslag', alpha=0.95)
    ax1.set_ylabel('Hartslag (BPM)', color=KLEUREN_PALETTE['hr'], fontweight='bold', fontsize=11)
    ax1.tick_params(axis='y', labelcolor=KLEUREN_PALETTE['hr'])

    # 3. Snelheidslijn
    ax2 = ax1.twinx()
    ax2.plot(df['elapsed_minutes'], df['speed_kmh'], color=KLEUREN_PALETTE['speed'], linestyle='-', linewidth=1.1, alpha=0.65, label='Snelheid')
    ax2.set_ylabel('Snelheid (km/u)', color=KLEUREN_PALETTE['speed'], fontweight='bold', fontsize=11)
    ax2.tick_params(axis='y', labelcolor=KLEUREN_PALETTE['speed'])

    # 4. Afwerking
    format_time_axis(ax1)
    ax1.set_ylim(min_hr - 8, max_hr + 16)
    ax2.set_ylim(0, df['speed_kmh'].max() * 1.25)
    plt.title('RACE OPBOUW: RUN - BIKE - RUN', fontsize=13, fontweight='bold', pad=25, color=KLEUREN_PALETTE['text'])
    
    ax1.grid(True, axis='y', linestyle=':', alpha=0.12, color='#FFFFFF')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return fig, ax1, ax2

def plot_run_comparison(df_active, bike_start, bike_eind):
    """Visualiseert de vergelijking tussen Run 1 en Run 2 met trendlijnen."""
    apply_triathlon_style()
    
    df_calc = df_active.copy()
    df_calc['speed_smooth'] = df_calc['speed_kmh'].rolling(window=20, min_periods=5).mean()
    df_calc['hr_smooth'] = df_calc['heart_rate'].rolling(window=20, min_periods=5).mean()

    run1 = df_calc[(df_calc['elapsed_minutes'] < bike_start) & 
                   (df_calc['speed_smooth'] >= 8) & (df_calc['speed_smooth'] <= 15) & 
                   (df_calc['hr_smooth'] > 140)].dropna()

    run2 = df_calc[(df_calc['elapsed_minutes'] > bike_eind) & 
                   (df_calc['speed_smooth'] >= 8) & (df_calc['speed_smooth'] <= 15) & 
                   (df_calc['hr_smooth'] > 140)].dropna()

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    color_run1, color_run2 = KLEUREN_PALETTE['hr'], KLEUREN_PALETTE['speed']

    ax.scatter(run1['speed_kmh'], run1['heart_rate'], color=color_run1, alpha=0.25, label='Run 1 (Fris)', s=25)
    ax.scatter(run2['speed_kmh'], run2['heart_rate'], color=color_run2, alpha=0.25, label='Run 2 (Vermoeid)', s=25)

    if not run1.empty and not run2.empty:
        z1 = np.polyfit(run1['speed_smooth'], run1['hr_smooth'], 1)
        z2 = np.polyfit(run2['speed_smooth'], run2['hr_smooth'], 1)

        x1_range = np.linspace(run1['speed_smooth'].min(), run1['speed_smooth'].max(), 100)
        x2_range = np.linspace(run2['speed_smooth'].min(), run2['speed_smooth'].max(), 100)

        ax.plot(x1_range, np.poly1d(z1)(x1_range), color=color_run1, linewidth=3.5, label='Trend Run 1')
        ax.plot(x2_range, np.poly1d(z2)(x2_range), color=color_run2, linewidth=3.5, label='Trend Run 2')

    ax.set_xlabel('Snelheid (km/u)', fontsize=12, fontweight='bold', labelpad=10, color=KLEUREN_PALETTE['muted'])
    ax.set_ylabel('Hartslag (BPM)', fontsize=12, fontweight='bold', labelpad=10, color=KLEUREN_PALETTE['muted'])
    plt.title('VERGELIJKING: RUN 1 VERSUS RUN 2', fontsize=15, fontweight='bold', pad=20, color=KLEUREN_PALETTE['text'])
    ax.legend(frameon=True, facecolor=KLEUREN_PALETTE['card_bg'], edgecolor='none', fontsize=10, loc='upper left')

    plt.tight_layout()
    return fig

def export_for_instagram(fig, filename, format_type='portrait', output_dir='../output'):
    """Slaat de grafiek op in het gewenste formaat."""
    os.makedirs(output_dir, exist_ok=True)
    
    dimensions = {
        'portrait': (10.8, 13.5),    # 4:5 (Instagram Post)
        'square': (10.8, 10.8),      # 1:1 (Vierkant)
        'story': (10.8, 19.2),       # 9:16 (Story/Reel)
        'landscape': (13.5, 7.59),   # 16:9 (Liggend / Breedbeeld)
    }
    
    if format_type in dimensions:
        width, height = dimensions[format_type]
        fig.set_size_inches(width, height)

    filepath = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.4, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Visualisatie opgeslagen: {filepath}")