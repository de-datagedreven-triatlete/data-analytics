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

from utils.config import ANALYSE, KLEUREN
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
# Triatlon
# ---------------------------------------------------------------------------
def teken_race_opbouw(df: pd.DataFrame, fases: list[dict], titel: str = "RACE OPBOUW: RUN - BIKE - RUN"):
    """Tekent hartslag en snelheid over de wedstrijd, met de fasen (run, wissel, fiets) erin aangegeven.

    Args:
        df: Activiteit met `verstreken_minuten`, `hartslag` en `snelheid_kmh`.
        fases: Lijst met per fase `naam`, `start` en `eind` (minuten), zie `detecteer_triatlonfasen`.
    """
    pas_stijl_toe()
    fig, ax_hartslag = plt.subplots(figsize=(10, 6), dpi=300)

    heeft_hartslag = df["hartslag"].notna().any()
    max_hr = df["hartslag"].max() if heeft_hartslag else 180
    min_hr = df["hartslag"].min() if heeft_hartslag else 100

    for fase in fases:
        ax_hartslag.axvline(fase["start"], color="#333745", linestyle=":", linewidth=1.0)
        midden = (fase["start"] + fase["eind"]) / 2
        ax_hartslag.text(midden, max_hr + 5, fase["naam"], ha="center", va="bottom",
                         fontweight="bold", color=KLEUREN["gedempt"], fontsize=9)
    if fases:
        ax_hartslag.axvline(fases[-1]["eind"], color="#333745", linestyle=":", linewidth=1.0)

    ax_hartslag.plot(df["verstreken_minuten"], df["hartslag"], color=KLEUREN["hartslag"], linewidth=1.8, alpha=0.95)
    ax_hartslag.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontweight="bold", fontsize=11)
    ax_hartslag.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])

    ax_snelheid = ax_hartslag.twinx()
    ax_snelheid.plot(df["verstreken_minuten"], df["snelheid_kmh"], color=KLEUREN["snelheid"], linewidth=1.1, alpha=0.65)
    ax_snelheid.set_ylabel("Snelheid (km/u)", color=KLEUREN["snelheid"], fontweight="bold", fontsize=11)
    ax_snelheid.tick_params(axis="y", labelcolor=KLEUREN["snelheid"])
    ax_snelheid.set_ylim(0, df["snelheid_kmh"].max() * 1.25)

    formatteer_tijdas(ax_hartslag)
    ax_hartslag.set_ylim(min_hr - 8, max_hr + 16)
    ax_hartslag.grid(True, axis="y", linestyle=":", alpha=0.12, color="#FFFFFF")
    ax_hartslag.spines["top"].set_visible(False)
    ax_snelheid.spines["top"].set_visible(False)
    ax_hartslag.set_title(titel, fontsize=13, fontweight="bold", pad=25, color=KLEUREN["tekst"])

    fig.tight_layout()
    return fig


def teken_run_vergelijking(run1: pd.DataFrame, run2: pd.DataFrame, titel: str = "VERGELIJKING: RUN 1 VERSUS RUN 2"):
    """Vergelijkt hartslag bij gelijke snelheid in run 1 (fris) en run 2 (vermoeid).

    Door de punten van elke run loopt een trendlijn (lineaire regressie op de gladgestreken waarden).
    Ligt de trendlijn van run 2 hoger, dan kostte dezelfde snelheid meer hartslag.

    Args:
        run1, run2: Resultaat van `selecteer_runs_voor_vergelijking`.
    """
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
# Wandeletappes
# ---------------------------------------------------------------------------
def teken_vam_dashboard(df: pd.DataFrame, venster_s: int = ANALYSE.vam_venster_s, etappe_nr: int | None = None):
    """Tekent klim-intensiteit (VAM) tegen hartslag, met het hoogteprofiel op de achtergrond.

    Args:
        df: Resultaat van `bereken_vam`.
        venster_s: Het tijdvenster (s) waarmee de VAM is berekend; alleen voor de legenda.
    """
    pas_stijl_toe()
    fig, ax_vam = plt.subplots(figsize=(13.33, 7.5), dpi=300)
    x = df["actieve_minuten"]

    hoogte_glad = df["hoogte"].rolling(window=ANALYSE.hoogte_glad_venster, center=True, min_periods=1).mean()
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
    if df["hartslag"].notna().any():
        ax_hartslag = ax_vam.twinx()
        ax_hartslag.spines["right"].set_position(("outward", 10))
        hartslag_glad = df["hartslag"].rolling(window=60, min_periods=1).mean()
        ax_hartslag.plot(x, hartslag_glad, color=KLEUREN["hartslag"], linewidth=1.8, label="Hartslag (bpm)", zorder=4)
        ax_hartslag.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontweight="bold")
        ax_hartslag.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
        ax_hartslag.set_ylim(int(hartslag_glad.min() - 15), int(hartslag_glad.max() + 15))
        gem_hartslag_tekst = f"{int(df['hartslag'].mean())} bpm"

    voeg_statistiekenblok_toe(ax_vam, [
        f"Totale stijging: +{int(df['totale_stijging_m'].iloc[-1])} m",
        f"Gem. VAM: {gem_vam} m/u",
        f"Max. VAM: {max_vam} m/u",
        f"Gem. hartslag: {gem_hartslag_tekst}",
    ])

    ax_vam.set_xlabel("Actieve tijd (minuten)", fontweight="bold", color=KLEUREN["tekst"])
    fig.suptitle(maak_titel("Klim-intensiteit (VAM) versus inspanning (hartslag)", etappe_nr),
                 fontsize=14, fontweight="bold", y=0.96, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig


def teken_helling_cadans(klim: pd.DataFrame, resultaat: dict, etappe_nr: int | None = None):
    """Tekent de cadans (stappen per minuut) tegen de helling, met regressielijn.

    Args:
        klim, resultaat: Uitkomst van `analyseer_helling_cadans`.
    """
    pas_stijl_toe()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_facecolor(KLEUREN["kaart"])

    spreiding = ax.scatter(klim["helling_pct"], klim["cadans"], c=klim["hoogte"], cmap="viridis",
                           alpha=0.6, edgecolors="none", s=25, label="Metingen")

    x = np.array([klim["helling_pct"].min(), klim["helling_pct"].max()])
    y = resultaat["intercept"] + resultaat["hellingscoefficient"] * x
    ax.plot(x, y, color=KLEUREN["hartslag"], linewidth=2.5,
            label=f"Trendlijn ({resultaat['hellingscoefficient']:+.2f} spm per +1%)")

    ax.set_xlabel("Helling (%)", fontweight="bold", color=KLEUREN["tekst"])
    ax.set_ylabel("Cadans (stappen per minuut)", fontweight="bold", color=KLEUREN["tekst"])
    ax.set_title(maak_titel("Terreinimpact: helling versus cadans", etappe_nr), fontweight="bold", pad=15, color=KLEUREN["tekst"])

    kleurbalk = fig.colorbar(spreiding, ax=ax)
    kleurbalk.set_label("Hoogte (m)", fontweight="bold", color=KLEUREN["tekst"])
    kleurbalk.ax.yaxis.set_tick_params(color=KLEUREN["tekst"])
    plt.setp(kleurbalk.ax.get_yticklabels(), color=KLEUREN["tekst"])

    voeg_statistiekenblok_toe(ax, [
        f"• Cadansverandering per +10% helling: {resultaat['verandering_per_10_pct']:+.1f} spm",
        f"• Gem. cadans bij klimmen: {int(resultaat['gemiddelde_cadans'])} spm",
        f"• Max. helling geanalyseerd: {resultaat['max_helling_pct']:.1f}%",
    ], x=0.03)

    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="lower left", **_LEGENDA)
    fig.tight_layout()
    return fig


def teken_herstelhartslag(df: pd.DataFrame, pauzes: pd.DataFrame, tijden: dict, etappe_nr: int | None = None):
    """Tekent de hartslag met de rustpauzes en per pauze de daling in de eerste minuut.

    Args:
        df: Activiteit met `verstreken_minuten`, `hartslag`, `hartslag_glad` en `hoogte`.
        pauzes: Resultaat van `bereken_herstelhartslag`.
        tijden: Resultaat van `bereken_beweegtijden`.
    """
    pas_stijl_toe()
    fig, ax_hartslag = plt.subplots(figsize=(13.33, 7.5), dpi=300)
    x = df["verstreken_minuten"]

    if df["hoogte"].notna().any():
        teken_hoogteprofiel(ax_hartslag, x, df["hoogte"].interpolate().ffill().bfill(), ruimte_boven=0.4)

    ax_hartslag.plot(x, df["hartslag_glad"], color=KLEUREN["hartslag"], linewidth=1.8, label="Hartslag (bpm)", zorder=12)
    ax_hartslag.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontweight="bold")
    ax_hartslag.tick_params(axis="y", labelcolor=KLEUREN["hartslag"])
    ax_hartslag.set_ylim(int(df["hartslag"].min() - 10), int(df["hartslag"].max() + 15))

    for nummer, pauze in enumerate(pauzes.itertuples()):
        ax_hartslag.axvspan(pauze.start_minuut, pauze.start_minuut + pauze.duur_min, color=KLEUREN["rust"],
                            alpha=0.18, zorder=11, label="Rustpauze" if nummer == 0 else None)
        ax_hartslag.annotate(
            f"-{int(pauze.herstel_bpm)} bpm",
            xy=(pauze.start_minuut, pauze.hartslag_start), xytext=(pauze.start_minuut, pauze.hartslag_start + 6),
            fontsize=8.5, fontweight="bold", color=KLEUREN["tekst"], ha="center", zorder=13,
            arrowprops=dict(arrowstyle="->", color=KLEUREN["rust"], lw=1.2),
        )

    voeg_statistiekenblok_toe(ax_hartslag, [
        f"Bewogen tijd:    {formatteer_duur(tijden['bewogen_s'])}",
        f"Verstreken tijd: {formatteer_duur(tijden['verstreken_s'])}",
        f"Gem. hartslag:   {int(df['hartslag'].mean())} bpm",
        f"Max. hartslag:   {int(df['hartslag'].max())} bpm",
    ], x=0.03, y=0.93, vaste_breedte=True, fontsize=9.5)

    ax_hartslag.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), facecolor=KLEUREN["blok"],
                       edgecolor=KLEUREN["blokrand"], labelcolor="white", fontsize=9)
    ax_hartslag.set_xlabel("Verstreken tijd (minuten)", fontweight="bold", color=KLEUREN["tekst"])
    fig.suptitle(maak_titel("Herstelhartslag (HRR)", etappe_nr), fontsize=14, fontweight="bold", y=0.96, color=KLEUREN["tekst"])
    fig.tight_layout()
    return fig


def teken_beweegratio(raster: pd.DataFrame, resultaat: dict, etappe_nr: int | None = None):
    """Tekent de snelheid met de rustpauzes, plus een cirkeldiagram van actief versus stilstand.

    Args:
        raster, resultaat: Uitkomst van `bereken_beweegratio`.
    """
    pas_stijl_toe()
    fig = plt.figure(figsize=(14, 7))
    kolommen = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.15)
    ax_hoofd = fig.add_subplot(kolommen[0])
    ax_cirkel = fig.add_subplot(kolommen[1])
    x = raster["verstreken_minuten"]

    if raster["hoogte"].notna().any():
        teken_hoogteprofiel(ax_hoofd, x, raster["hoogte"], toon_as=True, ruimte_boven=0.6)

    max_snelheid = max(raster["snelheid_glad_kmh"].max() * 1.15, 5.0)
    ax_hoofd.fill_between(x, 0, max_snelheid, where=raster["is_pauze"], color=KLEUREN["rust"], alpha=0.28,
                          step="post", label="Rustpauze")
    ax_hoofd.plot(x, raster["snelheid_glad_kmh"], color=KLEUREN["snelheid"], linewidth=2.0, label="Snelheid (km/u)")

    ax_hoofd.set_xlabel("Verstreken tijd (minuten)", color="white", fontsize=11, fontweight="bold", labelpad=8)
    ax_hoofd.set_ylabel("Snelheid (km/u)", color=KLEUREN["snelheid"], fontsize=11, fontweight="bold", labelpad=8)
    ax_hoofd.set_ylim(bottom=-0.1, top=max_snelheid)
    ax_hoofd.grid(True, linestyle="--", alpha=0.15)
    ax_hoofd.set_title(maak_titel("Beweegratio (actief versus stilstand)", etappe_nr),
                       color="white", fontsize=13, fontweight="bold", pad=15)

    ratio = resultaat["beweegratio_pct"]
    voeg_statistiekenblok_toe(ax_hoofd, [
        f"Totale tijd:   {formatteer_duur(resultaat['totale_tijd_s'], 'kort')}",
        f"In beweging:  {formatteer_duur(resultaat['beweging_s'], 'kort')} ({ratio:.1f}%)",
        f"Stilstand:    {formatteer_duur(resultaat['stilstand_s'], 'kort')} ({100 - ratio:.1f}%)",
    ], x=0.03, y=0.93, vaste_breedte=True, fontsize=9.5)
    ax_hoofd.legend(loc="upper right", facecolor=KLEUREN["blok"], edgecolor=KLEUREN["blokrand"], labelcolor="white", fontsize=9)

    ax_cirkel.pie([ratio, max(0, 100 - ratio)], colors=[KLEUREN["goed"], KLEUREN["rust"]], startangle=90,
                  counterclock=False, wedgeprops=dict(width=0.26, edgecolor=KLEUREN["achtergrond"], linewidth=3))
    ax_cirkel.text(0, 0, f"{int(round(ratio))}%\nActief", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax_cirkel.set_title("Verhouding actief / stilstand", color="white", fontsize=11, fontweight="bold", pad=12)

    fig.subplots_adjust(left=0.07, right=0.93, wspace=0.25)
    return fig


def teken_cardiac_drift(actief: pd.DataFrame, resultaat: dict, etappe_nr: int | None = None):
    """Tekent snelheid en hartslag over de actieve tijd, met de efficiëntiefactor van beide helften.

    De efficiëntiefactor (EF) is de hartslag per km/u gecorrigeerde snelheid. Is de EF in de
    tweede helft hoger, dan kostte dezelfde inspanning meer hartslag: cardiac drift.

    Args:
        actief, resultaat: Uitkomst van `bereken_cardiac_drift`.
    """
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


def teken_hoogtefactor(df: pd.DataFrame, etappe_nr: int | None = None):
    """Vergelijkt de afstand op een vlakke kaart (2D) met de afstand over het echte terrein (3D).

    Boven: 2D- en 3D-afstand met het hoogteprofiel. Onder: de cumulatieve extra meters door de steilte.

    Args:
        df: Resultaat van `bereken_3d_afstand`.
    """
    pas_stijl_toe()
    fig, (ax_boven, ax_onder) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                              gridspec_kw={"height_ratios": [2, 1]})
    x = df["afstand_2d_km"]

    teken_hoogteprofiel(ax_boven, x, df["hoogte_glad"], toon_as=True, ruimte_boven=0.3)
    ax_boven.plot(x, x, color=KLEUREN["gedempt"], linestyle="--", linewidth=2, label="2D-afstand (platte kaart)")
    ax_boven.plot(x, df["afstand_3d_km"], color=KLEUREN["snelheid"], linewidth=2.5, label="3D-afstand (werkelijk)")
    ax_boven.set_ylabel("Berekende afstand (km)", color="white", fontsize=10, fontweight="bold")
    ax_boven.set_title(maak_titel("De hoogtefactor (2D plat versus 3D werkelijk)", etappe_nr),
                       color="white", fontsize=12, fontweight="bold", pad=15)
    ax_boven.grid(False)

    totaal_2d = df["afstand_2d_km"].iloc[-1]
    totaal_3d = df["afstand_3d_km"].iloc[-1]
    extra_m = df["cumulatief_extra_m"].iloc[-1]
    extra_pct = extra_m / (totaal_2d * 1000) * 100 if totaal_2d > 0 else 0.0
    voeg_statistiekenblok_toe(ax_boven, [
        f"Totale 2D-afstand: {totaal_2d:.2f} km",
        f"Totale 3D-afstand: {totaal_3d:.2f} km",
        "---------------------------",
        f"Extra steilte-meters: +{extra_m:.0f} m (+{extra_pct:.1f}%)",
    ])
    ax_boven.legend(loc="lower right", facecolor="#121212", edgecolor="#333333", labelcolor="white")

    ax_onder.plot(x, df["cumulatief_extra_m"], color=KLEUREN["hartslag"], linewidth=2, label="Cumulatieve extra 3D-meters")
    ax_onder.fill_between(x, 0, df["cumulatief_extra_m"], color=KLEUREN["hartslag"], alpha=0.25)
    ax_onder.set_xlabel("Kaartafstand (km)", color="white", fontsize=10, fontweight="bold")
    ax_onder.set_ylabel("Extra meters (m)", color="white", fontsize=10, fontweight="bold")
    ax_onder.grid(False)
    ax_onder.legend(loc="lower right", facecolor="#121212", edgecolor="#333333", labelcolor="white")

    fig.tight_layout()
    return fig


def teken_hrv_readiness(df_etappe: pd.DataFrame, herstel: pd.DataFrame, etappe_nr: int):
    """Tekent de hartslag over de etappe en daaronder het verloop van HRV en readiness per dag.

    De dag van de etappe wordt in de onderste grafieken gemarkeerd.

    Args:
        df_etappe: Activiteit met `hartslag`, `hoogte` en (optioneel) `afstand_m`.
        herstel: Resultaat van `lees_herstelgegevens` (`dag`, `hrv_rmssd`, `readiness_score`).
        etappe_nr: Nummer van de dag/etappe die wordt gemarkeerd.
    """
    pas_stijl_toe()
    fig = plt.figure(figsize=(12, 9))
    raster = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1], width_ratios=[1, 1], figure=fig)

    # --- Hartslag en hoogte tijdens de etappe ---
    ax_hoofd = fig.add_subplot(raster[0, :])
    if df_etappe["afstand_m"].notna().any():
        x, x_label = df_etappe["afstand_m"] / 1000.0, "Afstand (km)"
    else:
        x, x_label = np.arange(len(df_etappe)), "Meting (nummer)"

    if df_etappe["hoogte"].notna().any():
        teken_hoogteprofiel(ax_hoofd, x, df_etappe["hoogte"].interpolate().ffill().bfill(), toon_as=True, ruimte_boven=0.1)

    hartslag_glad = df_etappe["hartslag"].rolling(window=60, min_periods=1).mean()
    ax_hoofd.plot(x, hartslag_glad, color=KLEUREN["hartslag"], linewidth=2.2, label="Hartslag (60 s gem.)", zorder=4)

    dag_nu = herstel[herstel["dag"] == etappe_nr]
    regels = [f"Gem. hartslag: {df_etappe['hartslag'].mean():.0f} bpm", f"Max. hartslag: {df_etappe['hartslag'].max():.0f} bpm"]
    if not dag_nu.empty:
        regels += [f"HRV: {dag_nu['hrv_rmssd'].iloc[0]:.0f} ms", f"Readiness: {dag_nu['readiness_score'].iloc[0]:.0f}"]
    voeg_statistiekenblok_toe(ax_hoofd, regels, y=0.90, fontsize=9)

    ax_hoofd.set_xlabel(x_label, color="white", fontsize=10, fontweight="bold")
    ax_hoofd.set_ylabel("Hartslag (bpm)", color=KLEUREN["hartslag"], fontsize=10, fontweight="bold")
    ax_hoofd.set_title(maak_titel("Hartslag, HRV en readiness", etappe_nr), color="white", fontsize=11, fontweight="bold", pad=12)
    ax_hoofd.grid(False)
    ax_hoofd.legend(loc="lower left", **{**_LEGENDA, "fontsize": 8})

    dagen = [f"Dag {dag}" for dag in herstel["dag"]]
    positie = herstel.index[herstel["dag"] == etappe_nr]

    # --- HRV-verloop ---
    ax_hrv = fig.add_subplot(raster[1, 0])
    ax_hrv.plot(dagen, herstel["hrv_rmssd"], color=KLEUREN["snelheid"], marker="o", linewidth=2.5, markersize=8, label="HRV-verloop")
    if len(positie) > 0:
        i = positie[0]
        is_laagste = herstel["hrv_rmssd"].idxmin() == i
        ax_hrv.scatter(dagen[i], herstel["hrv_rmssd"].iloc[i], color=KLEUREN["hartslag"], s=120, zorder=5,
                       edgecolor="white", linewidth=1.5, label=f"Dag {etappe_nr}" + (" (laagste)" if is_laagste else ""))
    ax_hrv.set_title("HRV-status", color="white", fontsize=11, fontweight="bold", pad=10)
    ax_hrv.set_ylabel("Milliseconden (ms)", color="white", fontsize=9, fontweight="bold")
    ax_hrv.grid(False)
    ax_hrv.legend(loc="lower left", **{**_LEGENDA, "fontsize": 8})

    # --- Readiness ---
    ax_readiness = fig.add_subplot(raster[1, 1])
    staven = ax_readiness.bar(dagen, herstel["readiness_score"], color=KLEUREN["snelheid"], width=0.5)
    if len(positie) > 0:
        staven[positie[0]].set_color(KLEUREN["hartslag"])
    for staaf in staven:
        ax_readiness.text(staaf.get_x() + staaf.get_width() / 2, staaf.get_height() + 2, f"{int(staaf.get_height())}",
                          ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")
    ax_readiness.set_title("Training readiness", color="white", fontsize=11, fontweight="bold", pad=10)
    ax_readiness.set_ylabel("Score (0-100)", color="white", fontsize=9, fontweight="bold")
    ax_readiness.set_ylim(0, 105)
    ax_readiness.grid(False)

    fig.tight_layout()
    return fig


def teken_cadanstrend(gefilterd: pd.DataFrame, profiel: pd.DataFrame, baseline_spm: float, etappe_nr: int,
                      terrein: str = "klim", grens_pct: float = ANALYSE.terrein_grens_pct,
                      rooster_m: float = ANALYSE.terrein_rooster_m):
    """Tekent het verloop van de cadans op één terreintype, vergeleken met de baseline van week 1.

    Args:
        gefilterd: Resultaat van `bereken_cadanstrend` op het gefilterde terrein.
        profiel: Het volledige terreinprofiel (`maak_terreinprofiel`), voor het hoogteprofiel.
        baseline_spm: Gemiddelde klimcadans in de baseline-periode (stappen per minuut).
        terrein: 'klim', 'afdaling' of 'vlak'; bepaalt de titel.
    """
    titels = {
        "klim": f"Cadanstrend bij klimmen vanaf +{grens_pct:.1f}%",
        "afdaling": f"Cadanstrend bij dalen vanaf -{grens_pct:.1f}%",
        "vlak": f"Cadanstrend op vlak terrein (-{grens_pct:.1f}% tot +{grens_pct:.1f}%)",
    }
    pas_stijl_toe()
    fig, ax = plt.subplots(figsize=(12, 6))

    teken_hoogteprofiel(ax, profiel["afstand_km"], profiel["hoogte"], toon_as=True, ruimte_boven=0.3)

    if gefilterd.empty:
        ax.text(0.5, 0.5, "Geen metingen voor dit terreintype", color="white", ha="center", va="center", transform=ax.transAxes)
    else:
        afstand = gefilterd["afstand_km"].to_numpy()
        cadans = gefilterd["cadans_glad"].to_numpy().copy()
        cadans[np.r_[False, np.diff(afstand) > 1.5 * rooster_m / 1000.0]] = np.nan  # Lijn onderbreken tussen losse stukken

        gemiddelde = gefilterd["cadans"].mean()
        ax.plot(afstand, cadans, color=KLEUREN["snelheid"], linewidth=2.2, label=f"Etappe {etappe_nr}", zorder=4)
        ax.axhline(baseline_spm, color=KLEUREN["hartslag"], linestyle="--", linewidth=1.8, label="Baseline week 1", zorder=3)
        voeg_statistiekenblok_toe(ax, [
            f"Gem. cadans etappe {etappe_nr}: {gemiddelde:.1f} spm",
            f"Baseline week 1: {baseline_spm:.1f} spm",
            "---------------------------",
            f"Verschil: {gemiddelde - baseline_spm:+.1f} spm",
        ])

    ax.set_xlabel("Afstand (km)", color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cadans (stappen per minuut)", color=KLEUREN["snelheid"], fontsize=10, fontweight="bold")
    ax.set_title(maak_titel(titels[terrein], None), color="white", fontsize=11, fontweight="bold", pad=12)
    ax.grid(False)
    ax.legend(loc="lower right", **{**_LEGENDA, "fontsize": 8})

    fig.tight_layout()
    return fig


def teken_segment_profiel(
    df_klim: pd.DataFrame,
    df_rest: pd.DataFrame,
    df_segment: pd.DataFrame,
    statistieken: dict,
    doel_afstand_m: float,
    etappe_nr: int | None = None
):
    """Tekent het hoogteprofiel en markeert het steilste segment van de beklimming.
   
    Args:
        df_klim, df_rest, df_segment, statistieken: Output van `bereken_steilste_segment`.
        doel_afstand_m: Lengte van het gezochte segment in meters (voor de labels).
    """
    pas_stijl_toe()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    # 1. Bepaal het absolute minimum en maximum van de *gehele* etappe
    laagste_punt = min(df_klim["hoogte"].min(), df_rest["hoogte"].min())
    hoogste_punt = max(df_klim["hoogte"].max(), df_rest["hoogte"].max())
   
    # 2. Stel de bodem van de grafiek in (met 100m speling eronder)
    min_elev = max(0, laagste_punt - 100)
    max_elev = hoogste_punt + 150

    # Lijnen plotten
    ax.plot(df_klim["afstand_km"], df_klim["hoogte"], color="#888888", linewidth=1.5, label="Beklimming")
   
    if not df_segment.empty:
        ax.plot(df_segment["afstand_km"], df_segment["hoogte"], color=KLEUREN["snelheid"], linewidth=2.5, label=f"Steilste {int(doel_afstand_m)}m")
        # De arcering vult nu netjes door tot de nieuwe, absolute bodem (min_elev)
        ax.fill_between(df_segment["afstand_km"], df_segment["hoogte"], min_elev, color=KLEUREN["snelheid"], alpha=0.25)
   
    ax.plot(df_rest["afstand_km"], df_rest["hoogte"], color="#333333", linewidth=1.5, linestyle="--", label="Rest van etappe (buiten analyse)")
    ax.fill_between(df_rest["afstand_km"], df_rest["hoogte"], min_elev, color="#222222", alpha=0.4, hatch="//")

    # Opmaak assen
    ax.set_xlabel("Afstand (km)", color="white", fontweight="bold", labelpad=10)
    ax.set_ylabel("Hoogte (m)", color="white", fontweight="bold", labelpad=10)
    ax.set_title(maak_titel(f"Beklimming - Steilste {int(doel_afstand_m)}m", etappe_nr), color="white", fontsize=11, fontweight="bold", pad=15)
   
    # Y-as dynamisch schalen op basis van de gehele etappe
    ax.set_ylim(min_elev, max_elev)
    ax.grid(False)

    # Statistiekenblok
    stats_klim = statistieken["klim"]
    stats_seg = statistieken["segment"]
   
    voeg_statistiekenblok_toe(ax, [
        f"Totale klim vs. Steilste {int(doel_afstand_m)}m:",
        f"Duur:         {stats_klim['duur_min']:.1f}m vs {stats_seg['duur_min']:.1f}m",
        f"VAM:          {stats_klim['vam']} vs {stats_seg['vam']} m/u",
        f"Gem. HR:      {stats_klim['hartslag']} vs {stats_seg['hartslag']} bpm",
        f"Gem. helling: {stats_klim['gem_helling_pct']:.1f}% vs {stats_seg['gem_helling_pct']:.1f}%"
    ], x=0.02, y=0.95, vaste_breedte=True, fontsize=8.5)

    ax.legend(loc="lower right", facecolor="#111111", edgecolor=KLEUREN["blokrand"], labelcolor="white", fontsize=9)
   
    fig.tight_layout()
    return fig


