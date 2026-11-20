import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import AtleetConfig, KLEUREN

atleet = AtleetConfig()

# 1. Paginaconfiguratie
st.set_page_config(page_title="Triathlon App", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Triathlon Applicatie 🏊‍♀️ 🚴‍♀️ 🏃‍♀️")

KLEUREN_SPORT = {
    'Zwemmen': KLEUREN["zwemmen"],
    'Hardlopen': KLEUREN["hardlopen"],
    'Fietsen': KLEUREN["fietsen"]
}

def apply_chart_layout(fig):
    fig.update_layout(
        font_family="Inter",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=KLEUREN["tekst"],
        xaxis=dict(gridcolor=KLEUREN["blokrand"]),
        yaxis=dict(gridcolor=KLEUREN["blokrand"])
    )
    return fig

def format_uren_naar_hhmmss(uren_decimaal):
    if pd.isna(uren_decimaal) or uren_decimaal == 0:
        return "0:00:00"
    teken = "-" if uren_decimaal < 0 else ""
    uren_abs = abs(uren_decimaal)
    uren = int(uren_abs)
    resterende_seconden = round((uren_abs - uren) * 3600)
    minuten = int(resterende_seconden // 60)
    seconden = int(resterende_seconden % 60)
    if minuten == 60:
        uren += 1
        minuten = 0
    return f"{teken}{uren}:{minuten:02d}:{seconden:02d}"

def format_delta_uren(uren_decimaal):
    if pd.isna(uren_decimaal) or round(uren_decimaal * 3600) == 0:
        return "0:00:00"
    prefix = "+" if uren_decimaal > 0 else ""
    return f"{prefix}{format_uren_naar_hhmmss(uren_decimaal)}"

def sec_naar_minsec(seconden):
        m = int(seconden // 60)
        s = int(seconden % 60)
        return f"{m}:{s:02d}"

def parse_tempo(tempo_str):
    if not isinstance(tempo_str, str): return 300 
    m, s = map(int, tempo_str.split(':'))
    return m * 60 + s
    
def formateer_slider_tijd(minuten):
    h = int(minuten // 60)
    m = int(minuten % 60)
    return f"{h:02d}:{m:02d}"
    
def format_komma_decimaal(getal, decimalen=2):
    if pd.isna(getal):
        return "0,00"
    return f"{getal:.{decimalen}f}".replace('.', ',')

def format_duizendtallen(val):
    try:
        if pd.isna(val):
            return "-"
        return f"{float(val):,.0f}".replace(',', '.')
    except Exception:
        return str(val)
    
def calculate_pmc(data):
    df_daily = data.groupby(data['datum'].dt.date)['tss'].sum().reset_index()
    df_daily.columns = ['datum', 'tss']
    df_daily['datum'] = pd.to_datetime(df_daily['datum'])

    if df_daily.empty:
        return pd.DataFrame()

    all_days = pd.date_range(start=df_daily['datum'].min(), end=df_daily['datum'].max())
    pmc_df = pd.DataFrame({'datum': all_days}).merge(df_daily, on='datum', how='left').fillna({'tss': 0})

    pmc_df['Conditie (CTL)'] = pmc_df['tss'].ewm(span=42, adjust=False).mean()
    pmc_df['Vermoeidheid (ATL)'] = pmc_df['tss'].ewm(span=7, adjust=False).mean()
    pmc_df['Vorm (TSB)'] = pmc_df['Conditie (CTL)'].shift(1) - pmc_df['Vermoeidheid (ATL)'].shift(1)

    acute_load = pmc_df['tss'].rolling(window=7).sum()
    chronic_load = pmc_df['tss'].rolling(window=28).sum() / 4
    pmc_df['ACWR'] = acute_load / chronic_load

    pmc_df['ctl_str'] = pmc_df['Conditie (CTL)'].apply(format_komma_decimaal)
    pmc_df['atl_str'] = pmc_df['Vermoeidheid (ATL)'].apply(format_komma_decimaal)
    pmc_df['tsb_str'] = pmc_df['Vorm (TSB)'].apply(format_komma_decimaal)
    pmc_df['tss_str'] = pmc_df['tss'].round(0).astype(int).astype(str)
    return pmc_df

# --- DATA INLADEN ---
@st.cache_data
def load_data():
    df = pd.read_parquet("garmin_data.parquet")
    df['datum'] = pd.to_datetime(df['datum'])
    df['week_start'] = df['datum'].dt.to_period('W').apply(lambda r: r.start_time)
    df['week_label'] = "KW" + df['week_start'].dt.isocalendar().week.astype(str) + ": " + df['week_start'].dt.strftime('%d-%m-%Y')
    df['afstand_km'] = df['afstand_meters'] / 1000
    df['tijd_uren'] = df['tijd_seconden'] / 3600

    sport_mapping = {
        'running': 'Hardlopen',
        'cycling': 'Fietsen',
        'swimming': 'Zwemmen',
        'hiking': 'Hiken',
        'walking': 'Wandelen',
        'alpine_skiing': 'Skiën',
        'backcountry_skiing': 'Skiën',
        'cross_country_skiing': 'Langlaufen'
    }
    df['sport_nl'] = df['sport'].map(sport_mapping).fillna('Overig')
    df['is_triathlon'] = df['sport'].isin(['running', 'cycling', 'swimming'])
    return df

df_all = load_data()
df = df_all[df_all['is_triathlon']].copy()

pmc_data_volledig = calculate_pmc(df_all)
if not pmc_data_volledig.empty:
        laatste_rij = pmc_data_volledig.iloc[-1]
        actueel_ctl = laatste_rij['Conditie (CTL)']
        actueel_atl = laatste_rij['Vermoeidheid (ATL)']
        actueel_tsb = laatste_rij['Vorm (TSB)']
        actueel_acwr = laatste_rij['ACWR']
else:
    actueel_ctl, actueel_atl, actueel_tsb, actueel_acwr = 0, 0, 0, 0

ftp = atleet.ftp
drempel_sec_km = parse_tempo(getattr(atleet, 'drempeltempo', "5:46"))
css_sec_100m = 100 / atleet.CSS_ZWEMMEN if hasattr(atleet, 'CSS_ZWEMMEN') else 130

# --- TABBLADEN AANMAKEN ---
tab1, tab2 = st.tabs(["📊 Analyse", "🤖 AI Coach"])

# --- TABBLAD ANALYSE ---
with tab1:
    if not df_all.empty:
        huidige_week_start = df_all['week_start'].max()
        vorige_week_start = huidige_week_start - pd.Timedelta(days=7)

        df_huidig = df[df['week_start'] == huidige_week_start]
        df_vorig = df[df['week_start'] == vorige_week_start]

        tijd_huidig = df_huidig['tijd_uren'].sum()
        tijd_vorig = df_vorig['tijd_uren'].sum()
        delta_tijd = tijd_huidig - tijd_vorig

        tss_huidig = df_huidig['tss'].sum()
        tss_vorig = df_vorig['tss'].sum()
        delta_tss = tss_huidig - tss_vorig

        sessies_huidig = len(df_huidig)
        sessies_vorig = len(df_vorig)
        delta_sessies = sessies_huidig - sessies_vorig

        kw_nummer = huidige_week_start.isocalendar().week
        st.subheader(f"Status KW{kw_nummer} (t.o.v. vorige week)")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                label="Aantal Sessies",
                value=f"{sessies_huidig}",
                delta=f"{delta_sessies:+d}" if delta_sessies != 0 else "0"
            )
        
        with kpi2:
            st.metric(
                label="Totale Trainingstijd",
                value=f"{format_uren_naar_hhmmss(tijd_huidig)} uur",
                delta=f"{format_delta_uren(delta_tijd)} uur"
            )
        with kpi3:
            help_tekst_TSS = """TSS (Training Stress Score) is de maatstaf voor de fysiologische zwaarte van je training. Het combineert je trainingsduur met de intensiteit:
- Beginners: Trainen doorgaans met 200 tot 350 TSS per week om een stabiele basis te leggen zonder blessurerisico.
- Gevorderden: Zitten vaak tussen de 400 en 600 TSS per week.
- Wedstrijdsporters: Kunnen pieken naar 700 tot 1.000+ TSS per week, mits dit wordt afgewisseld met geplande herstelweken."""
            st.metric(
                label="Training Stress Score",
                value=f"{round(tss_huidig)}",
                delta=f"{round(delta_tss):+d}" if delta_tss != 0 else "0",
                help=help_tekst_TSS)

        with kpi4:
            if actueel_tsb > 25:
                advies_label, advies_kleur = "Rust- / detrainingsfase", KLEUREN["tekst"]
            elif actueel_tsb >= 5:
                advies_label, advies_kleur = "Frisheid & taper / wedstrijdfase", KLEUREN["zwemmen"]
            elif actueel_tsb >= -10:
                advies_label, advies_kleur = "Herstel- of overgangsfase", KLEUREN["hardlopen"]
            elif actueel_tsb >= -30:
                advies_label, advies_kleur = "Productieve trainingsfase", KLEUREN["fietsen"]
            else:
                advies_label, advies_kleur = "Zware opbouw / hoog risico", KLEUREN["blokrand"]

            help_tekst_TSB = """TSB (Training Stress Balance) is je Conditie (CTL) minus je Vermoeidheid (ATL):
- Boven +25: Je bent uitgerust, maar bij een langdurig hoge waarde loop je het risico dat je conditie weer afneemt door te weinig trainingsprikkels.
- +5 tot +25: Vermoeidheid zakt sneller weg dan je conditie (bijvoorbeeld tijdens een taper). Dit is de zone waarin je fris aan een belangrijke wedstrijd wilt starten.
- -10 tot +5: Vermoeidheid en conditie zijn in balans. Blijf hier niet te lang hangen zonder prikkel, anders stagneert je vooruitgang.
- -10 tot -30: Je bouwt conditie op terwijl je gecontroleerd vermoeidheid opstapelt. Dit is gebruikelijk tijdens een opbouw- of belastingsblok.
- Onder -30: Je vermoeidheid ligt veel hoger dan je conditie. Dit komt voor bij zware trainingsblokken, maar vergroot het risico op blessures, ziekte of overbelasting."""
            st.metric(
                label="Vorm", 
                value=f"{round(actueel_tsb)}", 
                help=help_tekst_TSB)
            st.markdown(f"<div style='margin-top:-15px; color:{advies_kleur}; font-size:14px; font-weight:600;'>{advies_label}</div>", unsafe_allow_html=True)

        col_links, col_rechts = st.columns([1.2, 1])

        with col_links:
            st.subheader("⚡ Acute:Chronic Workload Ratio")
            if pd.isna(actueel_acwr) or actueel_acwr == 0:
                acwr_kleur, acwr_status = "#9ca3af", "Onvoldoende data"
            elif actueel_acwr < 0.8:
                acwr_kleur, acwr_status = "#fbbf24", "Rust- / herstelperiode"
            elif actueel_acwr <= 1.3:
                acwr_kleur, acwr_status = "#10b981", "Productieve trainingsfase"
            elif actueel_acwr <= 1.5:
                acwr_kleur, acwr_status = "#f59e0b", "Zware trainingsfase"
            else:
                acwr_kleur, acwr_status ="#ef4444", "Gevarenzone"

            pct_breedte = min(100, max(0, (actueel_acwr / 2.0) * 100)) if not pd.isna(actueel_acwr) else 0
            weergave_waarde = format_komma_decimaal(actueel_acwr) if not pd.isna(actueel_acwr) else "-"

            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 5px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-size: 24px; font-weight: bold; color: {acwr_kleur};">{weergave_waarde}</span>
                    <span style="font-size: 14px; color: {acwr_kleur}; align-self: flex-end;">{acwr_status}</span>
                </div>
                <div style="width: 100%; background-color: #374151; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="width: {pct_breedte}%; background-color: {acwr_kleur}; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"ℹ️ Wat betekent mijn score van {weergave_waarde}?"):
                st.write("De ACWR (Acute:Chronic Workload Ratio) vergelijkt je vermoeidheid van de **afgelopen 7 dagen** met je opgebouwde onditie van de **afgelopen 28 dagen**.")
                st.write("""
                - **Onder 0,8:** Relatieve onderbelasting of rust-/herstelperiode ten opzichte van wat je gewend bent.
                - **0,8 - 1,3:** Geleidelijke opbouw of onderhoud in lijn met je basisniveau; traditioneel gezien als een relatief veilige bandbreedte.
                - **1,3 - 1,5:** Snelle piek in belasting, wat de kans op overbelasting of blessures kan vergroten
                - **Boven 1,5:** Gevarenzone! De belasting stijgt te snel. Het risico op blessures is nu statistisch erg hoog.
                """)

            st.write("")
            col_titel, col_filter = st.columns([1, 1])
            with col_titel:
                st.subheader("📅 Trainingskalender")
            with col_filter:
                unieke_weken_lijst = df[['week_start', 'week_label']].drop_duplicates().sort_values('week_start', ascending=False)['week_label'].tolist()
                kalender_keuze = st.selectbox("Periode:", ["Laatste 4 weken"] + unieke_weken_lijst, label_visibility="collapsed")

            if kalender_keuze == "Laatste 4 weken":
                start_4w = huidige_week_start - pd.Timedelta(weeks=3)
                df_kalender = df[df['week_start'] >= start_4w].copy()
            else:
                df_kalender = df[df['week_label'] == kalender_keuze].copy()

            if not df_kalender.empty:
                dagen_map = {0: 'Ma', 1: 'Di', 2: 'Wo', 3: 'Do', 4: 'Vr', 5: 'Za', 6: 'Zo'}
                df_kalender['Dag_Nummer'] = df_kalender['datum'].dt.dayofweek
                df_kalender['Dag'] = df_kalender['Dag_Nummer'].map(dagen_map)
                df_kalender['Week_Kort'] = "KW " + df_kalender['week_start'].dt.isocalendar().week.astype(str)

                dag_totaal = df_kalender.groupby(['Week_Kort', 'week_start', 'Dag', 'Dag_Nummer'])['tijd_uren'].sum().reset_index()

                sport_tekst = df_kalender.groupby(['Week_Kort', 'week_start', 'Dag'])['sport_nl'].apply(lambda x: ', '.join(x.unique())).reset_index()
                sport_tekst.rename(columns={'sport_nl': 'alle_sporten'}, inplace=True)

                langste_sport = df_kalender.sort_values('tijd_uren', ascending=False).drop_duplicates(subset=['Week_Kort', 'week_start', 'Dag'])
                langste_sport = langste_sport[['Week_Kort', 'week_start', 'Dag', 'sport_nl']].rename(columns={'sport_nl': 'Primaire_Sport'})

                kalender_data = dag_totaal.merge(sport_tekst, on=['Week_Kort', 'week_start', 'Dag']).merge(langste_sport, on=['Week_Kort', 'week_start', 'Dag'])
                kalender_data = kalender_data.sort_values('week_start')
                kalender_data['tijd_weergave'] = kalender_data['tijd_uren'].apply(format_uren_naar_hhmmss)

                fig_kalender = px.scatter(
                    kalender_data,
                    x='Dag',
                    y='Week_Kort',
                    size='tijd_uren',
                    color='Primaire_Sport',
                    color_discrete_map=KLEUREN_SPORT,
                    category_orders={"Dag": ['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo']},
                    custom_data=['tijd_weergave', 'alle_sporten']
                )
                fig_kalender.update_traces(
                    hovertemplate='<b>%{x}</b><br>Activiteit(en): %{customdata[1]}<br>Totale Tijd: %{customdata[0]}<extra></extra>',
                    marker=dict(line=dict(width=1, color=KLEUREN["tekst"]), opacity=0.9)
                )
                fig_kalender.update_xaxes(
                    type='category',
                    categoryorder='array',
                    categoryarray=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'],
                    tickvals=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'],
                    range=[-0.5, 6.5]
                )
                fig_kalender.update_layout(
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=False,
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(showgrid=True, gridcolor=KLEUREN["blokrand"], zeroline=False),
                    xaxis=dict(showgrid=False, zeroline=False)
                )
                fig_kalender = apply_chart_layout(fig_kalender)
                st.plotly_chart(fig_kalender, use_container_width=True)

        with col_rechts:
            with col_rechts:
                c_hist_titel, c_hist_info = st.columns([6, 1])
                with c_hist_titel:
                    st.subheader("📝 Historische Activiteiten")
                with c_hist_info:
                    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                    with st.popover("ℹ️", help="Wat betekent de TSS-score per training?"):
                        st.markdown("**Betekenis TSS-score & Hersteltijd**")
                        st.markdown("""
| TSS-score | Belastingsfase | Verwachte hersteltijd |
| :--- | :--- | :--- |
| **< 150** | Lage belasting | Herstel is doorgaans de volgende dag compleet. |
| **150 – 300** | Gemiddelde belasting | Je ervaart lichte vermoeidheid, met volledig herstel binnen twee dagen. |
| **300 – 450** | Hoge belasting | Je spieren zijn merkbaar vermoeid en hebben twee tot drie dagen herstel nodig. |
| **> 450** | Zeer hoge belasting | Dit komt voor bij zware wedstrijden of marathons en vraagt meerdere dagen rust. |
                """)
            alle_activiteiten = df.sort_values(by='datum', ascending=False)

            if not alle_activiteiten.empty:
                with st.container(height=500, border=False):
                    for _, row in alle_activiteiten.iterrows():
                        sport = row['sport_nl']
                        icon = "🏃" if sport == "Hardlopen" else "🚴" if sport == "Fietsen" else "🏊"

                        duur = format_uren_naar_hhmmss(row['tijd_uren'])
                        tss_val = f"{format_duizendtallen(row['tss'])} TSS" if pd.notnull(row['tss']) and row['tss'] > 0 else "-"
                        datum_str = row['datum'].strftime('%d-%m-%Y')

                        with st.expander(f"{icon} **{sport}** | {datum_str} | {duur} | {tss_val}"):
                            r1_c1, r1_c2, r1_c3 = st.columns(3)

                            if sport == "Zwemmen":
                                afstand_m = row['afstand_km'] * 1000
                                r1_c1.metric("Afstand", f"{format_duizendtallen(afstand_m)} m")
                            else:
                                r1_c1.metric("Afstand", f"{format_komma_decimaal(row['afstand_km'], 2)} km")

                            r1_c2.metric("Tijd", duur)

                            if sport == "Hardlopen":
                                if row['afstand_km'] > 0:
                                    minuten_per_km = (row['tijd_seconden'] / 60) / row['afstand_km']
                                    tempo_str = format_uren_naar_hhmmss(minuten_per_km / 60).replace('0:', '', 1) + "/km"
                                    r1_c3.metric("Tempo", tempo_str)
                                else:
                                    r1_c3.metric("Tempo", "-")
                            elif sport == "Zwemmen":
                                if row['afstand_km'] > 0:
                                    sec_per_100m = (row['tijd_seconden'] / (row['afstand_km'] * 1000)) * 100
                                    m = int(sec_per_100m // 60)
                                    s = int(sec_per_100m % 60)
                                    r1_c3.metric("Tempo", f"{m}:{s:02d}/100m")
                                else:
                                    r1_c3.metric("Tempo", "-")
                            else:
                                snelheid = row['afstand_km'] / row['tijd_uren'] if row['tijd_uren'] > 0 else 0
                                r1_c3.metric("Gem. Snelheid", f"{format_komma_decimaal(snelheid, 1)} km/u")

                            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                            r2_c1, r2_c2, r2_c3 = st.columns(3)

                            hr_val = f"{row['hartslag_gem']:.0f} bpm" if pd.notnull(row.get('hartslag_gem')) else "-"
                            r2_c1.metric("Gem. HR", hr_val)

                            max_hr_val = f"{row['hartslag_max']:.0f} bpm" if 'hartslag_max' in df.columns and pd.notnull(row.get('hartslag_max')) else "-"
                            r2_c2.metric("Max. HR", max_hr_val)

                            cal_val = f"{format_duizendtallen(row.get('calorieen'))} kcal" if pd.notnull(row.get('calorieen')) else "-"
                            r2_c3.metric("Calorieën", cal_val)

    st.markdown("---")
    st.header("📊 Wekelijkse Totalen")

    unieke_weken_df = df[['week_start', 'week_label']].drop_duplicates().sort_values('week_start', ascending=False)
    alle_week_labels_nieuw_oud = unieke_weken_df['week_label'].tolist()
    standaard_10_weken = alle_week_labels_nieuw_oud[:10]

    geselecteerde_weken = st.multiselect(
        "Selecteer weken (standaard de laatste 10 weken):",
        options=alle_week_labels_nieuw_oud,
        default=standaard_10_weken
    )

    metric_keuze = st.radio("Toon activiteiten op basis van:", ["Tijd (uren)", "Afstand (km)"], horizontal=True)

    if geselecteerde_weken:
        gefilterde_weken_df = unieke_weken_df[unieke_weken_df['week_label'].isin(geselecteerde_weken)].sort_values('week_start', ascending=True)
        chronologische_labels = gefilterde_weken_df['week_label'].tolist()
        df_analyse = df[df['week_label'].isin(chronologische_labels)].copy()

        col_bar, col_pie = st.columns([2.2, 1])

        with col_bar:
            alle_sporten = ['Fietsen', 'Hardlopen', 'Zwemmen']
            grid = pd.MultiIndex.from_product(
                [gefilterde_weken_df['week_start'], alle_sporten],
                names=['week_start', 'sport_nl']
            ).to_frame(index=False).merge(gefilterde_weken_df, on='week_start')

            wekelijkse_som = df_analyse.groupby(['week_start', 'week_label', 'sport_nl'])[['afstand_km', 'tijd_uren']].sum().reset_index()
            wekelijkse_totalen = grid.merge(wekelijkse_som, on=['week_start', 'week_label', 'sport_nl'], how='left').fillna({'afstand_km': 0, 'tijd_uren': 0})
            wekelijkse_totalen = wekelijkse_totalen.sort_values('week_start')

            wekelijkse_totalen['tijd_weergave'] = wekelijkse_totalen['tijd_uren'].apply(format_uren_naar_hhmmss)
            wekelijkse_totalen['afstand_weergave'] = wekelijkse_totalen['afstand_km'].apply(format_komma_decimaal)

            y_col = 'tijd_uren' if metric_keuze == "Tijd (uren)" else 'afstand_km'
            y_label = 'Tijd (Uren)' if metric_keuze == "Tijd (uren)" else 'Afstand (km)'

            fig_bars = px.bar(
                wekelijkse_totalen,
                x='week_label',
                y=y_col,
                color='sport_nl',
                color_discrete_map=KLEUREN_SPORT,
                labels={'week_label': 'Week', y_col: y_label, 'sport_nl': ''},
                custom_data=['tijd_weergave', 'afstand_weergave']
            )
            if metric_keuze == "Tijd (uren)":
                fig_bars.update_traces(hovertemplate='<b>%{x}</b><br>Sport: %{data.name}<br>Tijd: %{customdata[0]}<extra></extra>')
            else:
                fig_bars.update_traces(hovertemplate='<b>%{x}</b><br>Sport: %{data.name}<br>Afstand: %{customdata[1]} km<extra></extra>')

            fig_bars.update_xaxes(
                type='category',
                categoryorder='array',
                categoryarray=chronologische_labels,
                tickangle=-45
            )
            fig_bars.update_layout(
                xaxis_title="",
                yaxis_title=y_label,
                margin=dict(t=20, b=95, l=0, r=0),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.32,
                    xanchor="center",
                    x=0.5,
                    title_text=""
                )
            )
            fig_bars = apply_chart_layout(fig_bars)
            st.plotly_chart(fig_bars, use_container_width=True)

        with col_pie:
            zones_dict = atleet.hr_zones
            max_z2 = round(zones_dict["Z2 Duur"][1])
            max_z3 = round(zones_dict["Z3 Tempo"][1])

            c_titel, c_info = st.columns([5, 1])
            with c_titel:
                st.markdown("<div style='font-weight: 600; margin-top: 8px;'>Polarized training (80/20-methode)</div>", unsafe_allow_html=True)
            with c_info:
                with st.popover("ℹ️", help="Wat betekend polarized training?"):
                    st.markdown("**Betekenis polarized training (80/20-methode)**")
                    st.markdown(f"""
| Intensiteitszone | Doelverdeling | Hartslag (BPM) | Trainingseffect |
| :--- | :--- | :--- | :--- |
| **Lage intensiteit** | ~80% van de tijd | ≤ {max_z2} bpm | Aerobe basis opbouwen en vetverbranding (Zone 1 & 2). |
| **Grijze zone** | Vermijden | {max_z2 + 1} – {max_z3} bpm | Wel vermoeidheid, maar te weinig prikkel voor topvorm (Zone 3). |
| **Hoge intensiteit** | ~20% van de tijd | > {max_z3} bpm | Verhogen van VO2-max en omslagpunt (Zone 4 & 5). |
                """)

            def categoriseer_polarized_hr(data):
                hr_df = data.dropna(subset=['hartslag_gem']).copy()

                def bepaal_pol_zone(hr):
                    if hr <= max_z2:
                        return 'Lage intensiteit'
                    elif hr <= max_z3:
                        return 'Grijze zone'
                    else:
                        return 'Hoge intensiteit'

                hr_df['Pol_Zone'] = hr_df['hartslag_gem'].apply(bepaal_pol_zone)
                return hr_df

            pol_zone_data = categoriseer_polarized_hr(df_analyse)

            if not pol_zone_data.empty:
                tijd_per_zone = pol_zone_data.groupby('Pol_Zone')['tijd_uren'].sum().reset_index()
                tijd_per_zone['tijd_weergave'] = tijd_per_zone['tijd_uren'].apply(format_uren_naar_hhmmss)

                kleuren_pol = {
                    'Lage intensiteit': KLEUREN["zwemmen"],
                    'Grijze zone': '#D8B4C8',
                    'Hoge intensiteit': KLEUREN["fietsen"]
                }

                fig_pol = go.Figure(data=[go.Pie(
                    labels=tijd_per_zone['Pol_Zone'],
                    values=tijd_per_zone['tijd_uren'],
                    hole=0.4,
                    customdata=tijd_per_zone['tijd_weergave'],
                    marker=dict(colors=[kleuren_pol.get(z, '#FFFFFF') for z in tijd_per_zone['Pol_Zone']]),
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate="Tijd: %{customdata}<extra></extra>"
                )])
                fig_pol.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=350)
                fig_pol = apply_chart_layout(fig_pol)
                st.plotly_chart(fig_pol, use_container_width=True)
            else:
                st.info("Geen hartslagdata beschikbaar voor deze selectie.")

        with st.expander("📋 Overzichtstabel"):
            pivot_tabel = wekelijkse_totalen.pivot(index='week_start', columns='sport_nl', values=y_col).fillna(0)
            pivot_tabel['Totaal'] = pivot_tabel.sum(axis=1)
            pivot_tabel = pivot_tabel.sort_index(ascending=False)
            for col in pivot_tabel.columns:
                if metric_keuze == "Tijd (uren)":
                    pivot_tabel[col] = pivot_tabel[col].apply(format_uren_naar_hhmmss)
                else:
                    pivot_tabel[col] = pivot_tabel[col].apply(format_komma_decimaal)
            pivot_tabel.index = pivot_tabel.index.map(lambda d: f"KW{d.isocalendar().week}: " + d.strftime('%d-%m-%Y'))
            st.dataframe(pivot_tabel, use_container_width=True)
    else:
        st.warning("Selecteer minimaal één week om de grafieken weer te geven.")

    st.markdown("---")
    col_pmc_titel, col_pmc_filter = st.columns([2, 1])
    with col_pmc_titel:
        st.header("📈 Prestatie Management Grafiek")
    with col_pmc_filter:
        pmc_periode = st.selectbox(
            "Tijdvak:",
            ["Laatste 3 maanden", "Laatste 6 maanden", "Alle beschikbare data"],
            index=1,
            label_visibility="collapsed"
        )
    
    if not pmc_data_volledig.empty:
        laatste_dag_str = pmc_data_volledig['datum'].max().strftime('%d-%m-%Y')
    
    if actueel_tsb > 25:
        status_titel = "Rust- / detrainingsfase"
        status_uitleg = f"Je bent volledig uitgerust (vorm: {actueel_tsb:+.0f}), omdat je vermoeidheid ({actueel_atl:.0f}) flink lager ligt dan je conditie ({actueel_ctl:.0f}). Bij een langdurig hoge waarde loop je het risico dat je conditie weer afneemt door te weinig trainingsprikkels."
        rand_kleur = KLEUREN["tekst"]
    elif actueel_tsb >= 5:
        status_titel = "Frisheid & taper / wedstrijdfase"
        status_uitleg = f"Je vermoeidheid ({actueel_atl:.0f}) zakt sneller weg dan je conditie ({actueel_ctl:.0f}), wat zorgt voor een vorm van {actueel_tsb:+.0f}. Dit is de optimale zone om fris en scherp aan een belangrijke wedstrijd te starten."
        rand_kleur = KLEUREN["zwemmen"]
    elif actueel_tsb >= -10:
        status_titel = "Herstel- of overgangsfase"
        status_uitleg = f"Je vermoeidheid ({actueel_atl:.0f}) en conditie ({actueel_ctl:.0f}) zijn in balans (vorm: {actueel_tsb:+.0f}). Blijf hier niet te lang hangen zonder nieuwe trainingsprikkel, anders stagneert je vooruitgang."
        rand_kleur = KLEUREN["hardlopen"]
    elif actueel_tsb >= -30:
        status_titel = "Productieve trainingsfase"
        status_uitleg = f"Je bouwt conditie ({actueel_ctl:.0f}) op terwijl je gecontroleerd vermoeidheid ({actueel_atl:.0f}) opstapelt, wat zorgt voor een vorm van {actueel_tsb:.0f}. Dit is gebruikelijk en wenselijk tijdens een opbouw- of belastingsblok."
        rand_kleur = KLEUREN["fietsen"]
    else:
        status_titel = "Zware opbouw / hoog risico"
        status_uitleg = f"Je vermoeidheid ({actueel_atl:.0f}) ligt veel hoger dan je conditie ({actueel_ctl:.0f}), waardoor je vorm diep negatief staat ({actueel_tsb:.0f}). Dit komt voor bij zware trainingsblokken, maar vergroot het risico op blessures, ziekte of overbelasting."
        rand_kleur = KLEUREN["blokrand"]

    st.markdown(f"""
    <div style="background-color: rgba(255,255,255,0.04); border-left: 4px solid {rand_kleur}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;">
        <div style="font-size: 16px; font-weight: 600; margin-bottom: 4px; color: {rand_kleur};">Status t/m {laatste_dag_str}: {status_titel}</div>
        <div style="font-size: 14px; color: #d1d5db; line-height: 1.5;">
            {status_uitleg}<br>
            <span style="font-size: 13px; color: #9ca3af;">
                <b>Huidige cijfers:</b> Conditie (CTL): <b>{actueel_ctl:.0f}</b> &nbsp;|&nbsp; Vermoeidheid (ATL): <b>{actueel_atl:.0f}</b> &nbsp;|&nbsp; Vorm (TSB): <b>{actueel_tsb:+.0f}</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Hoe lees ik deze grafiek?"):
        st.write("""
- **Blauwe lijn (Conditie / CTL):** Je gemiddelde trainingsbelasting over de afgelopen 42 dagen. Hoe hoger deze lijn kruipt, hoe meer trainingsarbeid je lichaam aankan.
- **Roze lijn (Vermoeidheid / ATL):** Je gemiddelde trainingsbelasting over de afgelopen 7 dagen. Schiet deze lijn ver boven de blauwe lijn uit? Dan ben je hard aan het trainen, maar stapel je ook vermoeidheid op.
- **Gele lijn (Vorm / TSB):** Je Conditie (CTL) minus je Vermoeidheid (ATL):
  - **Boven +25:** Rust- / detrainingsfase — Je bent uitgerust, maar bij een langdurig hoge waarde loop je het risico dat je conditie afneemt door te weinig prikkels.
  - **+5 tot +25:** Frisheid & taper / wedstrijdfase — Vermoeidheid zakt sneller weg dan je conditie; ideaal om fris aan een belangrijke wedstrijd te starten.
  - **-10 tot +5:** Herstel- of overgangsfase — Vermoeidheid en conditie zijn in balans. Blijf hier niet te lang hangen zonder prikkel, anders stagneert je vooruitgang.
  - **-10 tot -30:** Productieve trainingsfase — Je bouwt conditie op terwijl je gecontroleerd vermoeidheid opstapelt tijdens een opbouwblok.
  - **Onder -30:** Zware opbouw / hoog risico — Je vermoeidheid ligt veel hoger dan je conditie, wat het risico op blessures, ziekte of overbelasting vergroot.
- **Gekleurde stippen (Dagelijkse TSS):** Elke stip is een trainingsdag, gekleurd naar de hoofdactiviteit van die dag.
        """)


    if not pmc_data_volledig.empty:
        max_datum = pmc_data_volledig['datum'].max()
        if pmc_periode == "Laatste 3 maanden":
            pmc_data = pmc_data_volledig[pmc_data_volledig['datum'] >= (max_datum - pd.DateOffset(months=3))].copy()
        elif pmc_periode == "Laatste 6 maanden":
            pmc_data = pmc_data_volledig[pmc_data_volledig['datum'] >= (max_datum - pd.DateOffset(months=6))].copy()
        else:
            pmc_data = pmc_data_volledig.copy()


        df_all_dag = df_all.copy()
        df_all_dag['dag_datum'] = pd.to_datetime(df_all_dag['datum'].dt.date)
       
        primaire_dag_sport = df_all_dag.sort_values('tss', ascending=False).drop_duplicates(subset=['dag_datum'])
        primaire_dag_sport = primaire_dag_sport[['dag_datum', 'sport_nl']].rename(columns={'dag_datum': 'datum', 'sport_nl': 'Hoofdsport'})
       
        dag_sporten_tekst = df_all_dag.groupby('dag_datum')['sport_nl'].apply(lambda x: ', '.join(x.unique())).reset_index()
        dag_sporten_tekst.columns = ['datum', 'Alle_Sporten']


        dagen_tss = pmc_data[pmc_data['tss'] > 0].merge(primaire_dag_sport, on='datum', how='left').merge(dag_sporten_tekst, on='datum', how='left')
        dagen_tss['Hoofdsport'] = dagen_tss['Hoofdsport'].fillna('Overig')
        dagen_tss['Alle_Sporten'] = dagen_tss['Alle_Sporten'].fillna('Overig')


        kleuren_stippen = {
            'Zwemmen': KLEUREN["zwemmen"],
            'Hardlopen': KLEUREN["hardlopen"],
            'Fietsen': KLEUREN["fietsen"],
            'Hiken': '#D8B4C8',
            'Wandelen': '#D8B4C8',
            'Skiën': '#D8B4C8',
            'Langlaufen': '#D8B4C8',
            'Overig': '#D8B4C8'
        }


        fig_pmc = go.Figure()


        fig_pmc.add_trace(go.Scatter(
            x=pmc_data['datum'], y=pmc_data['Conditie (CTL)'],
            fill='tozeroy', mode='lines',
            name='Conditie',
            line=dict(color='#38BDF8', width=2.5),
            fillcolor='rgba(56, 189, 248, 0.15)',
            customdata=pmc_data[['ctl_str']],
            hovertemplate="Conditie (CTL): %{customdata[0]}<extra></extra>"
        ))


        fig_pmc.add_trace(go.Scatter(
            x=pmc_data['datum'], y=pmc_data['Vermoeidheid (ATL)'],
            mode='lines',
            name='Vermoeidheid',
            line=dict(color='#FF2A6D', width=2),
            customdata=pmc_data[['atl_str']],
            hovertemplate="Vermoeidheid (ATL): %{customdata[0]}<extra></extra>"
        ))


        fig_pmc.add_trace(go.Scatter(
            x=pmc_data['datum'], y=pmc_data['Vorm (TSB)'],
            mode='lines',
            name='Vorm',
            yaxis='y2',
            line=dict(color='#FBBF24', width=1.8),
            customdata=pmc_data[['tsb_str']],
            hovertemplate="Vorm (TSB): %{customdata[0]}<extra></extra>"
        ))


        stip_kleuren = [kleuren_stippen.get(s, '#9CA3AF') for s in dagen_tss['Hoofdsport']]
       
        fig_pmc.add_trace(go.Scatter(
            x=dagen_tss['datum'],
            y=dagen_tss['tss'],
            mode='markers',
            name='Dagelijkse TSS',
            marker=dict(
                color=stip_kleuren,
                size=6.5,
                opacity=0.95,
                line=dict(width=0.8, color='#FFFFFF')
            ),
            customdata=dagen_tss[['tss_str', 'Alle_Sporten']],
            hovertemplate="<b>%{customdata[1]}</b>: %{customdata[0]} TSS<extra></extra>"
        ))


        fig_pmc.add_hline(y=0, yref="y2", line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1)


        fig_pmc.update_layout(
            xaxis=dict(
                title="",
                tickformat="%d-%m-%Y",
                hoverformat="%d-%m-%Y",
                tickangle=0,
                nticks=8,
                range=[
                    pmc_data['datum'].min() - pd.Timedelta(days=1),
                    pmc_data['datum'].max() + pd.Timedelta(days=1)
                ]
            ),
            yaxis=dict(
                title="TSS",
                showgrid=True,
                gridcolor=KLEUREN["blokrand"],
                zeroline=False
            ),
            yaxis2=dict(
                title="Vorm (TSB)",
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=False
            ),
            hovermode="x unified",
            height=430,
            margin=dict(t=20, b=70, l=0, r=0),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5
            )
        )
        fig_pmc = apply_chart_layout(fig_pmc)
        st.plotly_chart(fig_pmc, use_container_width=True)

# --- TABBLAD AI COACH ---
with tab2:
    st.header("🤖 Vandaag's Training (AI Coach)")


    # --- HULPFUNCTIES ---
    def formateer_slider_tijd(minuten):
        h = int(minuten // 60)
        m = int(minuten % 60)
        return f"{h:02d}:{m:02d}"


    # --- 1. SLIMME STANDAARDWAARDEN BEPALEN O.B.V. DATA ---
    # Bepaal initieel herstel o.b.v. TSB
    if actueel_tsb > 5: init_herstel = 85
    elif actueel_tsb >= -10: init_herstel = 60
    elif actueel_tsb >= -25: init_herstel = 40
    else: init_herstel = 20


    # Bepaal initiële spierpijn o.b.v. gisteren (proxy: was gisteren een zware dag?)
    init_spierpijn = "Geen"
    if not df_all.empty:
        gisteren = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        df_gisteren = df_all[df_all['datum'].dt.normalize() == gisteren]
        if not df_gisteren.empty and df_gisteren['tss'].sum() > 100:
            init_spierpijn = "Matig"


    # --- 2. DE GEBRUIKERSINTERFACE (INPUT) ---
    st.subheader("1. Hoe voel je je vandaag?")
    col_ui1, col_ui2 = st.columns(2)
   
    with col_ui1:
        # Herstelniveau Slider
        herstelniveau = st.slider(
            "Herstelniveau (0 = uitgeput, 100 = topfit):",
            min_value=0, max_value=100, value=init_herstel, step=1,
            help="Wordt standaard geschat op basis van je actuele Vorm (TSB)."
        )
       
        # Kleur en label bepalen voor de UI
        if herstelniveau <= 30: herstel_kleur, herstel_label = "#ef4444", "🔴 Niet hersteld"
        elif herstelniveau <= 60: herstel_kleur, herstel_label = "#eab308", "🟡 Matig hersteld"
        else: herstel_kleur, herstel_label = "#10b981", "🟢 Goed hersteld"
       
        st.markdown(f"<div style='margin-top: -15px; color: {herstel_kleur}; font-weight: 600; font-size: 14px;'>{herstel_label}</div>", unsafe_allow_html=True)


    with col_ui2:
        # Spierpijn Selectie
        spierpijn = st.radio(
            "Mate van spierpijn:",
            options=["Geen", "Matig", "Veel"],
            index=["Geen", "Matig", "Veel"].index(init_spierpijn),
            horizontal=True
        )


    st.markdown("---")
    st.subheader("2. Jouw Beschikbaarheid")
    col_ui3, col_ui4 = st.columns(2)
    with col_ui3:
        tijd_opties = {formateer_slider_tijd(m): m for m in range(15, 375, 15)}
        geselecteerde_tijd_str = st.select_slider(
            "Hoeveel tijd heb je? (uu:mm):",
            options=list(tijd_opties.keys()),
            value="01:00"
        )
        beschikbare_tijd = tijd_opties[geselecteerde_tijd_str]
    with col_ui4:
        gewenste_sport = st.selectbox("Geplande Sport:", ["Fietsen", "Hardlopen", "Zwemmen"])


    # --- 3. DE AI COACH LOGICA (AANBEVELINGS-ENGINE) ---
    # Stap A: Basis categorie bepalen o.b.v. Herstel & Spierpijn
    if herstelniveau <= 30 or spierpijn == "Veel":
        advies_categorie = "Herstel / Uithoudingsvermogen"
        wetenschap_reden = "Een laag herstelniveau of veel spierpijn wijst op spierschade en vermoeidheid van het centrale zenuwstelsel. Intensiteit verhoogt nu enkel de vermoeidheid zonder adaptatie op te leveren. Actief herstel of rustige duur bevordert de doorbloeding."
    elif herstelniveau <= 60 or spierpijn == "Matig":
        advies_categorie = "Omslagpunt / Kracht"
        wetenschap_reden = "Je bent voldoende hersteld voor aerobe belasting, maar niet fris genoeg voor maximale neuromusculaire prikkels (VO2max). Werken op of net onder je omslagpunt is nu het meest effectief."
    else:
        advies_categorie = "VO2max"
        wetenschap_reden = "Je zenuwstelsel is fris en je spieren zijn hersteld. Dit is het fysiologische 'window of opportunity' om je maximale zuurstofopname en snelle spiervezels te prikkelen."


    # Stap B: Frequentie Check (Hebben we gisteren al zwaar getraind?)
    # We kijken naar de laatste 2 dagen voor de gekozen sport
    recente_zware_training = False
    if not df_all.empty:
        grens_datum = pd.Timestamp.now() - pd.Timedelta(days=2)
        recente_sessies = df_all[(df_all['sport_nl'] == gewenste_sport) & (df_all['datum'] >= grens_datum)]
       
        # Simpele proxy: als er een sessie was met een hoge TSS/uur ratio of hele hoge max HR
        if not recente_sessies.empty:
            for _, sessie in recente_sessies.iterrows():
                tss_per_uur = sessie['tss'] / sessie['tijd_uren'] if sessie['tijd_uren'] > 0 else 0
                if tss_per_uur > 75: # Indicatie van een intensieve (Omslagpunt/VO2max) sessie
                    recente_zware_training = True


    # Pas het advies aan als er recent al zwaar is getraind
    if recente_zware_training and advies_categorie in ["VO2max", "Omslagpunt / Kracht"]:
        advies_categorie = "Uithoudingsvermogen"
        wetenschap_reden += " **LET OP: Omdat je in de afgelopen 48 uur al een intensieve sessie hebt afgewerkt voor deze sport, schaalt de coach de intensiteit terug naar Uithoudingsvermogen om overtraining (tegen de 80/20 regel in) te voorkomen.**"


    # --- 4. WEERGAVE VAN HET ADVIES ---
    st.markdown("---")
    st.subheader("3. Coach Advies")
   
    st.info(f"🏆 **Aanbevolen Trainingsfocus:** {advies_categorie}")
    with st.expander("🔬 Wetenschappelijke onderbouwing", expanded=True):
        st.write(f"**Gekozen Sport:** {gewenste_sport} voor {formateer_slider_tijd(beschikbare_tijd)}")
        st.write(f"**Analyse:** {wetenschap_reden}")
        st.caption("De daadwerkelijke workout-generatie met blokken (zoals in JOIN) bouwen we in de volgende stap!")





