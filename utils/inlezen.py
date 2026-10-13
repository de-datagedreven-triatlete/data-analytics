"""Inlezen en opschonen van FIT- en CSV-bestanden.

Na `schoon_fit_data` hebben alle activiteiten dezelfde kolommen, ongeacht welk
horloge of welke firmware de data heeft opgenomen:

    tijd, hartslag (slagen/min), hoogte (m), afstand_m (m), snelheid_kmh (km/u),
    cadans, vermogen (W), breedtegraad (graden), lengtegraad (graden)

Alle berekeningen in `utils.berekeningen` gaan uit van deze kolomnamen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Per standaardkolom de kolommen uit het FIT-bestand, in volgorde van voorkeur.
# 'enhanced_*' bevat de nauwkeurigere waarde en gaat daarom voor.
FIT_KOLOMVOORKEUR = {
    "hartslag": ("heart_rate",),
    "hoogte": ("enhanced_altitude", "altitude"),
    "afstand_m": ("distance",),
    "snelheid_ms": ("enhanced_speed", "speed"),
    "cadans": ("cadence",),
    "vermogen": ("power",),
    "breedtegraad": ("position_lat",),
    "lengtegraad": ("position_long",),
}

# Waarden uit de samenvatting (session) van een FIT-bestand die we gebruiken
SESSIEVELDEN = (
    "total_ascent",
    "total_descent",
    "avg_heart_rate",
    "max_heart_rate",
    "total_distance",
    "total_timer_time",
)


def zoek_kolom(df: pd.DataFrame, kandidaten: tuple[str, ...]) -> str | None:
    """Geeft de eerste kandidaat die in `df` staat en minstens één waarde bevat.

    Geeft `None` terug als geen enkele kandidaat bruikbaar is.
    """
    for kandidaat in kandidaten:
        if kandidaat in df.columns and df[kandidaat].notna().any():
            return kandidaat
    return None


def lees_fit_bestand(pad: str | Path) -> pd.DataFrame:
    """Leest alle metingen (`record`-berichten) uit een .fit-bestand.

    De kolomnamen zijn nog die van het FIT-bestand (bijvoorbeeld `heart_rate`).
    Gebruik `schoon_fit_data` om ze naar het vaste formaat om te zetten.
    """
    import fitparse  # Hier geïmporteerd, zodat de rest van de module ook zonder fitparse werkt

    fitbestand = fitparse.FitFile(str(pad))
    metingen = [
        {veld.name: veld.value for veld in bericht}
        for bericht in fitbestand.get_messages("record")
    ]
    if not metingen:
        raise ValueError(f"Geen metingen gevonden in {pad}.")
    return pd.DataFrame(metingen)


def lees_sessiegegevens(pad: str | Path) -> dict:
    """Leest de samenvatting van de activiteit (`session`-bericht) uit een .fit-bestand.

    Dit zijn de totalen die het horloge zelf heeft berekend, zoals de totale
    stijging en daling. Velden die ontbreken, komen niet in het resultaat voor.
    """
    import fitparse

    fitbestand = fitparse.FitFile(str(pad))
    for bericht in fitbestand.get_messages("session"):
        waarden = {veld.name: veld.value for veld in bericht}
        return {naam: waarden[naam] for naam in SESSIEVELDEN if waarden.get(naam) is not None}
    return {}


def schoon_fit_data(df_ruw: pd.DataFrame, cadans_factor: float = 1.0) -> pd.DataFrame:
    """Zet ruwe FIT-metingen om naar het vaste formaat (zie moduledocstring).

    Stappen:
        1. Per grootheid de beste beschikbare FIT-kolom kiezen (`enhanced_*` gaat voor).
        2. Ontbrekende grootheden als lege kolom toevoegen (NaN).
        3. Snelheid omrekenen van m/s naar km/u; ontbreekt die, dan uit afstand en tijd afleiden.
        4. GPS-posities omrekenen van 'semicircles' naar graden.
        5. Cadans vermenigvuldigen met `cadans_factor` (2 bij wandelen: Garmin telt per been).
        6. Sorteren op tijd en dubbele tijdstippen verwijderen.

    Args:
        df_ruw: Resultaat van `lees_fit_bestand`.
        cadans_factor: Vermenigvuldigingsfactor voor de cadans.
    """
    if "timestamp" not in df_ruw.columns:
        raise ValueError("Het FIT-bestand bevat geen tijdstempels ('timestamp').")

    df = pd.DataFrame(index=df_ruw.index)
    df["tijd"] = pd.to_datetime(df_ruw["timestamp"]).astype("datetime64[ns]")

    for naam, kandidaten in FIT_KOLOMVOORKEUR.items():
        kolom = zoek_kolom(df_ruw, kandidaten)
        df[naam] = pd.to_numeric(df_ruw[kolom], errors="coerce") if kolom else np.nan

    # Snelheid in km/u
    if df["snelheid_ms"].notna().any():
        df["snelheid_kmh"] = df["snelheid_ms"] * 3.6
    elif df["afstand_m"].notna().any():
        seconden = df["tijd"].diff().dt.total_seconds().where(lambda s: s > 0)
        df["snelheid_kmh"] = df["afstand_m"].diff() / seconden * 3.6
    else:
        df["snelheid_kmh"] = np.nan
    df = df.drop(columns="snelheid_ms")

    # FIT slaat posities op als 'semicircles' (2^31 = 180 graden). Waarden boven 180 zijn dus
    # nog niet omgerekend.
    for kolom in ("breedtegraad", "lengtegraad"):
        if df[kolom].abs().max() > 180:
            df[kolom] = df[kolom] * (180.0 / 2**31)

    df["cadans"] = df["cadans"] * cadans_factor

    df = df.sort_values("tijd").drop_duplicates("tijd").reset_index(drop=True)
    return df


def bereken_verstreken_tijd(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt de kolom `verstreken_minuten` toe: minuten sinds de eerste meting."""
    df = df.copy()
    df["verstreken_minuten"] = (df["tijd"] - df["tijd"].min()).dt.total_seconds() / 60.0
    return df


def maak_tijdraster(df: pd.DataFrame, max_gat_s: int = 15) -> pd.DataFrame:
    """Zet de metingen om naar een aaneengesloten raster van één meting per seconde.

    Een horloge slaat bij een automatische pauze metingen over. Dit raster maakt
    die gaten zichtbaar:

        * Een meting geldt voor de seconden sinds de vorige meting, zolang het gat
          niet groter is dan `max_gat_s`.
        * Seconden in een groter gat tellen als stilstand (snelheid 0 km/u).
        * Hartslag en hoogte worden in een gat doorgetrokken vanaf de laatste bekende waarde.

    Returns:
        DataFrame met `tijd`, `verstreken_minuten`, `snelheid_kmh`, `hartslag` en `hoogte`.
    """
    bron = df.dropna(subset=["tijd"]).sort_values("tijd").drop_duplicates("tijd").reset_index(drop=True)
    bron["meettijd"] = bron["tijd"]
    bron["gat_s"] = bron["tijd"].diff().dt.total_seconds().fillna(0)

    raster = pd.DataFrame(
        {"tijd": pd.date_range(bron["tijd"].min(), bron["tijd"].max(), freq="1s").astype("datetime64[ns]")}
    )
    kolommen = ["tijd", "meettijd", "gat_s", "snelheid_kmh", "hartslag", "hoogte"]
    samengevoegd = pd.merge_asof(raster, bron[kolommen], on="tijd", direction="forward")

    gedekt = (samengevoegd["gat_s"] <= max_gat_s) | (samengevoegd["meettijd"] == samengevoegd["tijd"])
    samengevoegd["snelheid_kmh"] = samengevoegd["snelheid_kmh"].where(gedekt, 0.0).fillna(0.0)
    samengevoegd["hartslag"] = samengevoegd["hartslag"].where(gedekt).ffill().bfill()
    samengevoegd["hoogte"] = samengevoegd["hoogte"].where(gedekt).ffill().bfill()

    resultaat = samengevoegd[["tijd", "snelheid_kmh", "hartslag", "hoogte"]]
    return bereken_verstreken_tijd(resultaat)


def lees_herstelgegevens(pad: str | Path) -> pd.DataFrame:
    """Leest de dagelijkse HRV- en readiness-waarden uit een csv-bestand.

    Verwachte kolommen: `dag` (geheel getal), `hrv_rmssd` (ms) en `readiness_score` (0-100).
    """
    df = pd.read_csv(pad)
    ontbrekend = {"dag", "hrv_rmssd", "readiness_score"} - set(df.columns)
    if ontbrekend:
        raise ValueError(f"Kolommen ontbreken in {pad}: {sorted(ontbrekend)}")
    return df.sort_values("dag").reset_index(drop=True)
