"""Gedeelde hulpfuncties voor alle grafieken: stijl, opmaak en export.

De afzonderlijke grafieken staan in `utils.grafieken` en zijn opgebouwd uit deze functies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from utils.config import EXPORTFORMATEN, KLEUREN, PADEN


def pas_stijl_toe() -> None:
    """Stelt het uniforme donkere thema in voor alle grafieken."""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.facecolor": KLEUREN["achtergrond"],
            "axes.facecolor": KLEUREN["achtergrond"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": KLEUREN["raster"],
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": KLEUREN["gedempt"],
            "ytick.color": KLEUREN["gedempt"],
        }
    )


def maak_titel(onderwerp: str, etappe_nr: int | None = None) -> str:
    """Maakt een grafiektitel in hoofdletters, optioneel met het etappenummer."""
    voorvoegsel = f"ETAPPE {etappe_nr} PERFORMANCE" if etappe_nr else "ETAPPE PERFORMANCE"
    return f"{voorvoegsel}: {onderwerp}".upper()


def formatteer_duur(seconden: float, stijl: str = "klok") -> str:
    """Zet seconden om naar leesbare tekst.

    Args:
        stijl: 'klok' geeft 'uu:mm:ss' (of 'mm:ss' onder het uur), 'kort' geeft '2u 05m' (of '45m').
    """
    uren = int(seconden // 3600)
    minuten = int(seconden % 3600 // 60)
    rest = int(seconden % 60)
    if stijl == "kort":
        return f"{uren}u {minuten:02d}m" if uren > 0 else f"{minuten}m"
    if uren > 0:
        return f"{uren:02d}:{minuten:02d}:{rest:02d}"
    return f"{minuten:02d}:{rest:02d}"


def formatteer_tijdas(ax, label: str = "Tijd (minuten)") -> None:
    """Geeft de x-as een uniform label en hele getallen als waarden."""
    ax.set_xlabel(label, fontweight="bold", fontsize=10, color=KLEUREN["gedempt"], labelpad=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda waarde, _: f"{int(waarde)}"))


def voeg_statistiekenblok_toe(ax, regels: list[str], x: float = 0.02, y: float = 0.95,
                              vaste_breedte: bool = False, fontsize: float = 10) -> None:
    """Plaatst een blok met kerncijfers in een grafiek (standaard linksboven).

    Args:
        regels: Eén tekstregel per cijfer.
        x, y: Positie van de linkerbovenhoek, als fractie van de as (0 tot 1).
        vaste_breedte: Gebruik een lettertype met vaste breedte, zodat kolommen uitgelijnd blijven.
    """
    ax.text(
        x, y, "\n".join(regels),
        transform=ax.transAxes, fontsize=fontsize, color=KLEUREN["tekst"], fontweight="bold",
        fontfamily="monospace" if vaste_breedte else None,
        verticalalignment="top", zorder=20,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=KLEUREN["blok"], alpha=0.85, edgecolor=KLEUREN["blokrand"]),
    )


def teken_hoogteprofiel(ax, x, hoogte, toon_as: bool = False, ruimte_boven: float = 0.4):
    """Tekent een zacht hoogteprofiel op de achtergrond van een grafiek.

    Het profiel staat op een eigen (tweede) y-as achter de hoofdgrafiek.

    Args:
        ax: De hoofdas waarop de eigenlijke gegevens komen.
        x, hoogte: Waarden voor de x-as en de hoogte (m).
        toon_as: Toon de hoogteas rechts met label 'Hoogte (m)'. Standaard is de as verborgen.
        ruimte_boven: Extra ruimte boven het profiel, als fractie van het hoogteverschil.

    Returns:
        De as met het hoogteprofiel.
    """
    x = np.asarray(x)
    hoogte = np.asarray(hoogte, dtype=float)
    laagste, hoogste = np.nanmin(hoogte), np.nanmax(hoogte)
    bereik = hoogste - laagste if hoogste > laagste else 1.0

    as_hoogte = ax.twinx()
    as_hoogte.fill_between(x, hoogte, laagste, color=KLEUREN["hoogte"], alpha=0.18, zorder=1)
    as_hoogte.plot(x, hoogte, color=KLEUREN["hoogte"], linewidth=1.0, alpha=0.45, zorder=1)
    as_hoogte.set_ylim(laagste - bereik * 0.05, hoogste + bereik * ruimte_boven)
    as_hoogte.grid(False)

    if toon_as:
        as_hoogte.set_ylabel("Hoogte (m)", color=KLEUREN["gedempt"], fontsize=9)
        as_hoogte.tick_params(colors=KLEUREN["gedempt"], labelsize=8)
    else:
        as_hoogte.set_yticks([])

    # Het hoogteprofiel hoort achter de hoofdgrafiek te liggen
    as_hoogte.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    return as_hoogte


def exporteer_afbeelding(fig, bestandsnaam: str, formaat: str = "liggend",
                         uitvoermap: str | Path | None = None) -> Path:
    """Slaat een grafiek op als png-bestand in het gewenste formaat.

    Args:
        fig: De matplotlib-figuur.
        bestandsnaam: Naam zonder pad. Een eventueel achtervoegsel '.png' wordt genegeerd.
        formaat: 'staand' (4:5), 'vierkant' (1:1), 'verhaal' (9:16) of 'liggend' (16:9).
        uitvoermap: Doelmap; standaard `PADEN.uitvoer_map`.

    Returns:
        Het pad van het opgeslagen bestand.
    """
    if formaat not in EXPORTFORMATEN:
        raise ValueError(f"Onbekend formaat '{formaat}'; kies uit {sorted(EXPORTFORMATEN)}.")

    uitvoermap = Path(uitvoermap) if uitvoermap else PADEN.uitvoer_map
    uitvoermap.mkdir(parents=True, exist_ok=True)

    naam = bestandsnaam[:-4] if bestandsnaam.lower().endswith(".png") else bestandsnaam
    pad = uitvoermap / f"{naam}.png"

    fig.set_size_inches(*EXPORTFORMATEN[formaat])
    fig.savefig(pad, dpi=300, bbox_inches="tight", pad_inches=0.4,
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Afbeelding opgeslagen: {pad}")
    return pad
