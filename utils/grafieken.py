"""De grafieken van de analyses.
Elke functie tekent één grafiek op basis van gegevens die al zijn berekend met
`utils.berekeningen`, en geeft de matplotlib-figuur terug. Opslaan gaat met
`utils.visualisatie.exporteer_afbeelding`. In deze module staan dus geen berekeningen
van betekenis, alleen weergave.
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import KLEUREN
from utils.visualisatie import (
    formatteer_duur,
    formatteer_tijdas,
    maak_titel,
    pas_stijl_toe,
    teken_hoogteprofiel,
    voeg_statistiekenblok_toe,
)

_LEGENDA = dict(facecolor="#121212", edgecolor="#333333", labelcolor="white", fontsize=9)

# ---------------------------------------------------------------------------
# Triatlon & Multi-sport
# ---------------------------------------------------------------------------
def teken_race_opbouw(
    df: pd.DataFrame, 
    fases: list[dict], 
    titel: str = "RACE OPBOUW: RUN - BIKE - RUN"
) -> plt.Figure:
    """Tekent hartslag en snelheid over de wedstrijd, met de fasen (run, wissel, fiets) erin aangegeven."""
    pas_stijl_toe()
    fig, ax_hartslag = plt.subplots(figsize=(10, 6), dpi=300)
    
    heeft_hartslag = "hartslag" in df.columns and df["hartslag"].notna().any()
    max_hr = df["hartslag"].max() if heeft_hartslag else 180
    min_hr = df["hartslag"].min() if heeft_hartslag else 100
    
    for fase in fases:
        ax_hartslag.axvline(fase["start"], color="#333745", linestyle=":", linewidth=1.0)
        midden = (fase["start"] + fase["eind"]) / 2
        ax_hartslag.text(midden, max_hr + 5, fase["naam"], ha="center", va="bottom",
                         fontweight="bold", color=KLEUREN["gedempt"], fontsize=9)
    if fases:
        ax_hartslag.axvline(fases[-1]["eind"], color="#333745", linestyle=":", linewidth=1.0)
        
    if heeft_hartslag:
        ax_hartslag.plot(df["verstreken_minuten"], df["hartslag"], color=KLEUREN["hartslag"], linewidth=1.8, alpha=0.95)
        ax_hartslag.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontweight="bold", fontsize=11)
        ax_hartslag.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
        ax_hartslag.set_ylim(min_hr - 8, max_hr + 16)

    ax_snelheid = ax_hartslag.twinx()
    if "snelheid_kmh" in df.columns:
        ax_snelheid.plot(df["verstreken_minuten"], df["snelheid_kmh"], color=KLEUREN["snelheid"], linewidth=1.1, alpha=0.65)
        ax_snelheid.set_ylabel("Snelheid (km/u)", color=KLEUREN["snelheid"], fontweight="bold", fontsize=11)
        ax_snelheid.tick_params(axis="y", labelcolor=KLEUREN["snelheid"])
        ax_snelheid.set_ylim(0, df["snelheid_kmh"].max() * 1.25 if df["snelheid_kmh"].notna().any() else 40)

    formatteer_tijdas(ax_hartslag)
    ax_hartslag.grid(True, axis="y", linestyle=":", alpha=0.12, color="#FFFFFF")
    ax_hartslag.spines["top"].set_visible(False)
    ax_snelheid.spines["top"].set_visible(False)
    ax_hartslag.set_title(titel, fontsize=13, fontweight="bold", pad=25, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig


def teken_run_vergelijking(
    run1: pd.DataFrame, 
    run2: pd.DataFrame, 
    titel: str = "VERGELIJKING: RUN 1 VERSUS RUN 2"
) -> plt.Figure:
    """Vergelijkt hartslag bij gelijke snelheid in run 1 (fris) en run 2 (vermoeid)."""
    pas_stijl_toe()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    kleur1, kleur2 = KLEUREN["hartslag"], KLEUREN["snelheid"]
    
    ax.scatter(run1["snelheid_kmh"], run1["hartslag"], color=kleur1, alpha=0.25, label="Run 1 (fris)", s=25)
    ax.scatter(run2["snelheid_kmh"], run2["hartslag"], color=kleur2, alpha=0.25, label="Run 2 (vermoeid)", s=25)
    
    for run, kleur, label in ((run1, kleur1, "Trend run 1"), (run2, kleur2, "Trend run 2")):
        if len(run) > 1:
            coefficienten = np.polyfit(run["snelheid_glad_kmh"], run["hartslag_glad"], 1)
            x = np.linspace(run["snelheid_glad_kmh"].min(), run["snelheid_glad_kmh"].max(), 100)
            ax.plot(x, np.poly1d(coefficienten)(x), color=kleur, linewidth=3.5, label=label)
            
    ax.set_xlabel("Snelheid (km/u)", fontsize=12, fontweight="bold", labelpad=10, color=KLEUREN["gedempt"])
    ax.set_ylabel("Hartslag (bpm)", fontsize=12, fontweight="bold", labelpad=10, color=KLEUREN["gedempt"])
    ax.set_title(titel, fontsize=15, fontweight="bold", pad=20, color=KLEUREN["tekst"])
    ax.legend(frameon=True, facecolor=KLEUREN["kaart"], edgecolor="none", fontsize=10, loc="upper left")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Klimmen & Wandelen
# ---------------------------------------------------------------------------
def teken_vam_dashboard(
    df: pd.DataFrame, 
    venster_s: int = 300, 
    hoogte_glad_venster: int = 15,
    etappe_nr: int | None = None
) -> plt.Figure:
    """Tekent klim-intensiteit (VAM) tegen hartslag, met het hoogteprofiel op de achtergrond."""
    pas_stijl_toe()
    fig, ax_vam = plt.subplots(figsize=(13.33, 7.5), dpi=300)
    x = df["actieve_minuten"]
    hoogte_glad = df["hoogte"].rolling(window=hoogte_glad_venster, center=True, min_periods=1).mean()
    teken_hoogteprofiel(ax_vam, x, hoogte_glad, ruimte_boven=0.4)
    
    ax_vam.plot(x, df["vam_klim"], color=KLEUREN["snelheid"], linewidth=2.2, label=f"VAM ({venster_s} s gem.)", zorder=3)
    ax_vam.set_ylabel("VAM (m/u)", color=KLEUREN["snelheid"], fontweight="bold")
    ax_vam.tick_params(axis="y", labelcolor=KLEUREN["snelheid"])
    
    geldige_vam = df["vam_klim"].dropna()
    max_vam = int(geldige_vam.max()) if not geldige_vam.empty else 0
    gem_vam = int(geldige_vam.mean()) if not geldige_vam.empty else 0
    if max_vam > 0:
        ax_vam.set_ylim(0, max_vam * 1.25)
        
    gem_hartslag_tekst = "n.v.t."
    if "hartslag" in df.columns and df["hartslag"].notna().any():
        ax_hartslag = ax_vam.twinx()
        ax_hartslag.spines["right"].set_position(("outward", 10))
        hartslag_glad = df["hartslag"].rolling(window=60, min_periods=1).mean()
        ax_hartslag.plot(x, hartslag_glad, color=KLEUREN["hartslag"], linewidth=1.8, label="Hartslag (bpm)", zorder=4)
        ax_hartslag.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontweight="bold")
        ax_hartslag.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
        ax_hartslag.set_ylim(int(hartslag_glad.min() - 15), int(hartslag_glad.max() + 15))
        gem_hartslag_tekst = f"{int(df['hartslag'].mean())} bpm"
        
    voeg_statistiekenblok_toe(ax_vam, [
        f"Totale stijging: +{int(df['totale_stijging_m'].iloc[-1])} m" if "totale_stijging_m" in df.columns else "Stijging: N/A",
        f"Gem. VAM: {gem_vam} m/u",
        f"Max. VAM: {max_vam} m/u",
        f"Gem. hartslag: {gem_hartslag_tekst}",
    ])
    ax_vam.set_xlabel("Actieve tijd (minuten)", fontweight="bold", color=KLEUREN["tekst"])
    fig.suptitle(maak_titel("Klim-intensiteit (VAM) versus inspanning (hartslag)", etappe_nr),
                 fontsize=14, fontweight="bold", y=0.96, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig


def teken_cardiac_drift(actief: pd.DataFrame, resultaat: dict, etappe_nr: int | None = None) -> plt.Figure:
    """Tekent snelheid en hartslag over de actieve tijd, met de efficiëntiefactor van beide helften."""
    pas_stijl_toe()
    fig = plt.figure(figsize=(14, 7))
    kolommen = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.30)
    ax_snelheid = fig.add_subplot(kolommen[0])
    ax_hartslag = ax_snelheid.twinx()
    ax_ef = fig.add_subplot(kolommen[1])
    
    x = actief["actieve_minuten"]
    teken_hoogteprofiel(ax_snelheid, x, actief["hoogte_glad"], ruimte_boven=0.15)
    
    ax_snelheid.plot(x, actief["snelheid_glad_kmh"], color=KLEUREN["snelheid"], linewidth=1.8)
    ax_snelheid.set_xlabel("Actieve tijd (minuten)", color="white", fontsize=11, fontweight="bold", labelpad=8)
    ax_snelheid.set_ylabel("Snelheid (km/u)", color=KLEUREN["snelheid"], fontsize=11, fontweight="bold", labelpad=8)
    ax_snelheid.tick_params(axis="y", colors=KLEUREN["snelheid"])
    ax_snelheid.set_ylim(0, max(actief["snelheid_glad_kmh"].max() * 1.15, 5))
    ax_snelheid.grid(False)
    
    ax_hartslag.plot(x, actief["hartslag_glad"], color=KLEUREN["hartslag"], linewidth=1.8)
    ax_hartslag.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontsize=11, fontweight="bold", labelpad=8)
    ax_hartslag.tick_params(axis="y", colors=KLEUREN["hartslag"])
    ax_hartslag.set_ylim(actief["hartslag_glad"].min() * 0.85, actief["hartslag_glad"].max() * 1.10)
    ax_hartslag.grid(False)
    
    ax_snelheid.set_title(maak_titel("Cardiac drift en hoogte-impact", etappe_nr), color="white", fontsize=13, fontweight="bold", pad=15)
    drift = resultaat["drift_pct"]
    
    voeg_statistiekenblok_toe(ax_snelheid, [
        f"Gem. hartslag (helft 1): {resultaat['gem_hartslag_helft1']:.0f} bpm",
        f"Gem. hartslag (helft 2): {resultaat['gem_hartslag_helft2']:.0f} bpm",
        f"Gecorrigeerde drift:    {drift:+.1f}%",
    ], y=0.95, vaste_breedte=True, fontsize=9.5)
    
    drift_kleur = KLEUREN["hartslag"] if drift > 5.0 else KLEUREN["waarschuwing"] if drift > 0 else KLEUREN["snelheid"]
    ef = [resultaat["ef_helft1"], resultaat["ef_helft2"]]
    staven = ax_ef.bar(["Helft 1", "Helft 2"], ef, color=[KLEUREN["snelheid"], drift_kleur], width=0.55)
    for staaf, waarde in zip(staven, ef):
        ax_ef.text(staaf.get_x() + staaf.get_width() / 2, waarde, f"{waarde:.1f}", ha="center", va="bottom",
                   color="white", fontsize=10, fontweight="bold")
    ax_ef.set_ylim(0, max(ef) * 1.25)
    ax_ef.set_ylabel("Hartslag per km/u", color="white", fontsize=10, fontweight="bold")
    ax_ef.set_title(f"Efficiëntiefactor\nDrift {drift:+.1f}%", color="white", fontsize=11, fontweight="bold", pad=12)
    ax_ef.grid(False)
    fig.subplots_adjust(left=0.07, right=0.93, wspace=0.30)
    return fig

def teken_master_slide1_overzicht(
    df_master: pd.DataFrame, 
    totaal_stats: dict,
    df_zones: pd.DataFrame | None = None,
    titel: str = "MASTER ANALYSE: STUBAIER HÖHENWEG"
) -> plt.Figure:
    """
    Slide 1: Master Overzicht met:
    - Links Boven: Tabel met kerngetallen (inclusief Gem. Hartslag)
    - Rechts Boven: Donut Chart met titel direct erin/boven en uitgelijnde legenda
    - Onder: Hoogtemeters staafgrafiek per etappe (Stijging vs Daling)
    """
    pas_stijl_toe()
    fig = plt.figure(figsize=(10.8, 10.8), dpi=300)
    
    gs = gridspec.GridSpec(
        2, 2, 
        height_ratios=[1.0, 2.0], 
        width_ratios=[1.10, 0.90], 
        figure=fig, 
        hspace=0.28, 
        wspace=0.10
    )
    
    ax_tabel = fig.add_subplot(gs[0, 0])
    ax_donut = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[1, :])

    # --- 1. LINKS BOVEN: Tabel met Kerngetallen ---
    ax_tabel.axis("off")

    stijging_str = f"{totaal_stats['totale_stijging_m']:,.0f}".replace(",", ".")
    daling_str = f"{totaal_stats['totale_daling_m']:,.0f}".replace(",", ".")

    tabel_data = [
        ["Totale Afstand", f"{totaal_stats['totale_afstand_km']:.1f} km"],
        ["Totale Actieve Tijd", f"{totaal_stats['totale_duur_uur']:.1f} uur"],
        ["Totale Stijging", f"+{stijging_str} m"],
        ["Totale Daling", f"-{daling_str} m"],
    ]

    tabel = ax_tabel.table(
        cellText=tabel_data,
        colLabels=["Metriek", "Waarde"],
        loc="center",
        cellLoc="left"
    )
    tabel.auto_set_font_size(False)
    tabel.set_fontsize(8.5)
    tabel.scale(1.0, 1.35)
    
    for (row, col), cell in tabel.get_celld().items():
        cell.set_edgecolor(KLEUREN["blokrand"])
        if row == 0:
            cell.set_facecolor(KLEUREN["kaart"])
            cell.set_text_props(color=KLEUREN["snelheid"], fontweight="bold")
        else:
            cell.set_facecolor(KLEUREN["blok"])
            cell.set_text_props(color=KLEUREN["tekst"])

    # --- 2. RECHTS BOVEN: Donut Chart ---
    if df_zones is not None and not df_zones.empty:
        zone_betekenis = {
            "Z1 Herstel": "Z1 Herstel",
            "Z2 Duur": "Z2 Duur",
            "Z3 Tempo": "Z3 Tempo",
            "Z4 Drempel": "Z4 Drempel",
            "Z5 Maximaal": "Z5 Maximaal"
        }
        
        # Monochroom Cyaan verloop (van licht cyaan naar diep nachtblauw)
        verloop_kleuren = [
            "#80FAFF",  # Z1: Zeer licht cyaan
            "#00F5FF",  # Z2: Signature IJscyaan
            "#00B8D4",  # Z3: Midden cyaan
            "#007791",  # Z4: Donker cyaan
            "#004052",  # Z5: Diep nachtblauw
        ]
        
        actieve_zones = df_zones[df_zones["percentage"] > 0.1].copy()
        percentages = actieve_zones["percentage"].tolist()
        kleuren_subset = [verloop_kleuren[i % len(verloop_kleuren)] for i in actieve_zones.index]
        
        # 1. Teken de ring (zonder standaard teksten)
        wedges, _ = ax_donut.pie(
            percentages, 
            startangle=90,
            colors=kleuren_subset,
            wedgeprops=dict(width=0.38, edgecolor=KLEUREN["achtergrond"], linewidth=2)
        )

        # 2. Bepaal per segment de hoek en plaats schuin 'Z1', 'Z2', etc. IN de ring
        for i, wedge in enumerate(wedges):
            pct = percentages[i]
            if pct < 1.5:  # Sla hele kleine segmentjes over om overlapping te voorkomen
                continue
                
            # Middenhoek van het segment (in graden)
            theta = (wedge.theta1 + wedge.theta2) / 2.0
            
            # Straal (r = 0.81 valt exact in de gekleurde ring van width=0.38)
            r = 0.81 
            x = r * np.cos(np.deg2rad(theta))
            y = r * np.sin(np.deg2rad(theta))
            
            # Bepaal rotatiehoek zodat de tekst schuin meedraait met de cirkel
            rotation = theta - 90 if theta <= 180 else theta + 90
            
            # Kort label (bijv. "Z1", "Z2")
            zone_code = actieve_zones.iloc[i]["zone"].split(" ")[0]
            
            # Plaats de tekst schuin in de ring
            ax_donut.text(
                x, y, zone_code,
                ha="center", va="center",
                rotation=rotation,
                rotation_mode="anchor",
                color="#000000" if i < 2 else "#FFFFFF",  # Donkere tekst op licht cyaan, witte op donker
                fontweight="bold",
                fontsize=8
            )

        # 3. Legenda-labels opbouwen met zone-naam + exact percentage voor RECHTS
        legenda_labels = [
            f"{zone_betekenis.get(row['zone'], row['zone'])} ({row['percentage']:.0f}%)"
            for _, row in actieve_zones.iterrows()
        ]

        # 4. Legenda strak aan de RECHTERKANT van de Donut Chart plaatsen
        ax_donut.legend(
            wedges, 
            legenda_labels,
            loc="center left",
            bbox_to_anchor=(0.90, 0.5), 
            frameon=False,
            fontsize=8,
            labelcolor="white"
        )
        
        # Titel expliciet BOVEN de Donut Chart
        ax_donut.set_title("HARTSLAGZONES", color=KLEUREN["tekst"], fontweight="bold", fontsize=10, pad=10)
    else:
        ax_donut.axis("off")

    # --- 3. ONDERSTE DEEL: Hoogtemeters Staafgrafiek ---
    etappes = df_master["etappe"].to_numpy()
    x_indices = np.arange(len(etappes))
    breedte = 0.38

    ax_bars.bar(
        x_indices - breedte/2,
        df_master["stijging_m"],
        width=breedte,
        color=KLEUREN["snelheid"],  
        label="Stijging (m)",
        alpha=0.90
    )
    ax_bars.bar(
        x_indices + breedte/2,
        df_master["daling_m"],
        width=breedte,
        color=KLEUREN["hartslag"], 
        label="Daling (m)",
        alpha=0.90
    )
    
    ax_bars.set_ylabel("Hoogtemeters (m)", color=KLEUREN["tekst"], fontweight="bold", fontsize=10)
    ax_bars.set_xlabel("Etappe", color=KLEUREN["tekst"], fontweight="bold", fontsize=10)
    ax_bars.set_xticks(x_indices)
    ax_bars.set_xticklabels([f"E{e}" for e in etappes])
    ax_bars.set_xlim(-0.6, len(etappes) - 0.4)
    ax_bars.legend(loc="upper right", **_LEGENDA)
    ax_bars.grid(True, linestyle=":", alpha=0.15)
    
    fig.suptitle(titel, fontsize=13, fontweight="bold", y=0.98, color=KLEUREN["tekst"])
    return fig

def teken_master_slide2_vermoeidheid_pacing(
    df_master: pd.DataFrame,
    titel: str = "KLIM EFFICIENCY VS. CARDIAC DRIFT"
) -> plt.Figure:
    """Slide 2: Pacing & Klimvermoeidheid (Efficiency Index vs. Cardiac Drift per etappe)."""
    pas_stijl_toe()
    fig, ax1 = plt.subplots(figsize=(10.8, 10.8), dpi=300)
    
    etappes = df_master["etappe"].to_numpy()
    x_indices = np.arange(len(etappes))
    
    ei_values = df_master["efficiency_index"].copy()
    if "stijging_m" in df_master.columns:
        ei_values[df_master["stijging_m"] < 300] = np.nan

    ax1.plot(
        x_indices,
        ei_values,
        color=KLEUREN["cadans"],
        marker="o",
        linewidth=3.0,
        markersize=8,
        label="Efficiency Index (m/u klim per bpm)",
        zorder=4
    )
    ax1.set_ylabel("Klim Efficiency Index (VAM/HR)", color=KLEUREN["cadans"], fontweight="bold", fontsize=11)
    ax1.tick_params(axis="y", labelcolor=KLEUREN["cadans"])
    
    ax2 = ax1.twinx()
    ax2.axhline(0, color=KLEUREN["gedempt"], linestyle="--", linewidth=1.0, alpha=0.5)
    
    if "cardiac_drift_pct" in df_master.columns and df_master["cardiac_drift_pct"].notna().any():
        ax2.plot(
            x_indices,
            df_master["cardiac_drift_pct"],
            color=KLEUREN["hartslag"],
            marker="s",
            linestyle="--",
            linewidth=2.2,
            markersize=7,
            label="Cardiac Drift (%)",
            zorder=3
        )
        ax2.set_ylabel("Cardiac Drift (%)", color=KLEUREN["hartslag"], fontweight="bold", fontsize=11)
        ax2.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
        
        valid_drift = df_master["cardiac_drift_pct"].dropna()
        if not valid_drift.empty:
            max_drift_idx = valid_drift.idxmax()
            max_drift_val = valid_drift.loc[max_drift_idx]
            eff_val = ei_values.iloc[max_drift_idx]

    ax1.set_xlabel("Etappe", fontweight="bold", color=KLEUREN["tekst"], fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"E{e}" for e in etappes])
    ax1.grid(True, linestyle=":", alpha=0.15)
    
    fig.suptitle(titel, fontsize=13, fontweight="bold", y=0.98, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig


def teken_master_slide3_gezondheid_hrv(
    df_master: pd.DataFrame,
    titel: str = "HRV VS. KLIMCADANS"
) -> plt.Figure:
    """Slide 3: Impact van vermoeidheid op gezondheid & autonoom herstel (HRV vs. Klimcadans)."""
    pas_stijl_toe()
    fig, ax1 = plt.subplots(figsize=(10.8, 10.8), dpi=300)
    
    etappes = df_master["etappe"].to_numpy()
    x_indices = np.arange(len(etappes))
    
    if "hrv_rmssd" in df_master.columns and df_master["hrv_rmssd"].notna().any():
        ax1.plot(x_indices, df_master["hrv_rmssd"], color=KLEUREN["hartslag"], marker="o", linewidth=3.0, markersize=8, label="HRV RMSSD (ms)", zorder=4)
        ax1.set_ylabel("HRV (RMSSD in ms)", color=KLEUREN["hartslag"], fontweight="bold", fontsize=11)
        ax1.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
        
        hrv_valid = df_master["hrv_rmssd"].dropna()
        if len(hrv_valid) >= 2:
            hrv_start = hrv_valid.iloc[0]
            hrv_eind = hrv_valid.iloc[-1]

    # Klimcadans uitsluitend op klim-etappes
    if "klim_cadans" in df_master.columns and df_master["klim_cadans"].notna().any():
        cadans_values = df_master["klim_cadans"].copy()
        if "stijging_m" in df_master.columns:
            cadans_values[df_master["stijging_m"] < 300] = np.nan

        ax2 = ax1.twinx()
        ax2.plot(x_indices, cadans_values, color=KLEUREN["cadans"], marker="s", linestyle="--", linewidth=2.2, markersize=7, label="Klimcadans (spm)", zorder=3)
        ax2.set_ylabel("Klimcadans (spm)", color=KLEUREN["cadans"], fontweight="bold", fontsize=11)
        ax2.tick_params(axis="y", labelcolor=KLEUREN["cadans"])

    ax1.set_xlabel("Etappe", fontweight="bold", color=KLEUREN["tekst"], fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"E{e}" for e in etappes])
    ax1.grid(True, linestyle=":", alpha=0.15)
    
    fig.suptitle(titel, fontsize=13, fontweight="bold", y=0.98, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig

def teken_master_slide4_brandstof_glycogeen(
    df_master: pd.DataFrame,
    titel: str = "ENERGIE & GLYCOGEENVERBRUIK"
) -> plt.Figure:
    """
    Slide 4: Visualiseert het brandstofverbruik & glycogeen-uitputtingsmodel per etappe.
    Verwacht dat 'energie_kcal' en 'glycogeen_verbruik_g' al berekend zijn in df_master.
    """
    pas_stijl_toe()
    
    # Check of de benodigde kolommen aanwezig zijn
    if "energie_kcal" not in df_master.columns or "glycogeen_verbruik_g" not in df_master.columns:
        raise KeyError(
            "De kolommen 'energie_kcal' en/of 'glycogeen_verbruik_g' ontbreken in df_master. "
            "Voer eerst 'bereken_energie_en_glycogeen(df_master)' uit in utils.berekeningen."
        )
        
    fig, ax1 = plt.subplots(figsize=(10.8, 10.8), dpi=300)
    
    etappes = df_master["etappe"].to_numpy()
    x_indices = np.arange(len(etappes))
    breedte = 0.4

    # Calorieën als staven
    ax1.bar(
        x_indices, 
        df_master["energie_kcal"], 
        width=breedte, 
        color=KLEUREN["hartslag"], 
        alpha=0.75, 
        label="Energieverbruik (kcal)"
    )
    ax1.set_ylabel("Totaal Energieverbruik (kcal)", color=KLEUREN["hartslag"], fontweight="bold", fontsize=11)
    ax1.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
    
    # Glycogeenverbruik op 2e Y-as
    ax2 = ax1.twinx()
    ax2.plot(
        x_indices, 
        df_master["glycogeen_verbruik_g"], 
        color=KLEUREN["snelheid"], 
        marker="s", 
        linestyle="--",
        linewidth=2.2, 
        markersize=7, 
        label="Glycogeenverbruik (g carbs)"
    )
    ax2.set_ylabel("Geschat Glycogeenverbruik (gram carbs)", color=KLEUREN["snelheid"], fontweight="bold", fontsize=11)
    ax2.tick_params(axis="y", labelcolor=KLEUREN["snelheid"])
    
    # Samenvattende annotatie-box op basis van al aanwezige data
    tot_kcal = df_master["energie_kcal"].sum()
    tot_carbs_kg = df_master["glycogeen_verbruik_g"].sum() / 1000.0
    tot_kcal_str = f"{tot_kcal:,.0f}".replace(",", ".")
    tot_carbs_kg_str = f"{tot_carbs_kg:.2f}".replace(".", ",")

    ax1.annotate(
        f"Totale kcal: {tot_kcal_str}\nGlycogeen verbrand: {tot_carbs_kg_str} kg",
        xy=(0.03, 0.92),
        xycoords="axes fraction",
        fontsize=9.5, color=KLEUREN["tekst"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=KLEUREN["blok"], alpha=0.9, edgecolor=KLEUREN["blokrand"])
    )

    ax1.set_xlabel("Etappe", fontweight="bold", color=KLEUREN["tekst"], fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"E{e}" for e in etappes])
    ax1.grid(True, linestyle=":", alpha=0.15)
    
    fig.suptitle(titel, fontsize=13, fontweight="bold", y=0.98, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig