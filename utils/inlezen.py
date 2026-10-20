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
    "total_calories",
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
    import fitparse
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
        5. Cadans vermenigvuldigen met `cadans_factor` (2 bij wandelen/hardlopen: Garmin telt per been).
        6. Sorteren op tijd en dubbele tijdstippen verwijderen.
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

    # FIT slaat posities op als 'semicircles' (2^31 = 180 graden).
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
    Gaten groter dan `max_gat_s` tellen als stilstand (snelheid 0 km/u).
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

def laad_master_dataset(
    fit_map: str | Path,
    herstel_bestand: str | Path | None = None,
    naamfilter: str | None = None,
    cadans_factor: float = 1.0
) -> pd.DataFrame:
    """Generieke Master Inlezer:
    Leest een reeks FIT-bestanden in uit een map, sorteert ze op tijdstempel,
    nummering/etappes en koppelt optioneel dagelijkse herstelgegevens (HRV & Readiness).
    
    Args:
        fit_map: Map waarin de .fit bestanden staan.
        herstel_bestand: Optioneel pad naar herstel_hrv.csv.
        naamfilter: Optionele tekst waarop bestandsnamen gefilterd moeten worden.
        cadans_factor: Vermenigvuldigingsfactor voor de cadans (bijv. 2.0 bij lopen/wandelen).
    """
    alle_etappes = []
    
    # 1. Hersteldata optioneel inlezen indien opgegeven en aanwezig
    herstel_df = None
    if herstel_bestand and Path(herstel_bestand).exists():
        herstel_df = lees_herstelgegevens(herstel_bestand)

    # 2. FIT-bestanden ophalen en optioneel filteren
    fit_paden = sorted(list(Path(fit_map).glob("*.fit")))
    if naamfilter:
        fit_paden = [p for p in fit_paden if naamfilter in p.name]

    if not fit_paden:
        raise FileNotFoundError(f"Geen FIT-bestanden gevonden in {fit_map} (filter: '{naamfilter}').")

    # 3. Bestanden verwerken en op volgorde zetten
    for idx, pad in enumerate(fit_paden, start=1):
        # Bepaal etappenr: probeer te achterhalen uit bestandsnaam (bijv. "...etappe_1.fit"),
        # val anders terug op de chronologische index (1, 2, 3...)
        try:
            etappe_nr = int(pad.stem.split("etappe_")[-1])
        except (ValueError, IndexError):
            etappe_nr = idx

        df_ruw = lees_fit_bestand(pad)
        df_schoon = schoon_fit_data(df_ruw, cadans_factor=cadans_factor)
        df_schoon = bereken_verstreken_tijd(df_schoon)
        
        df_schoon["bron_bestand"] = pad.name
        df_schoon["etappe_nr"] = etappe_nr
        
        # 4. HRV & Readiness koppelen indien herstel_df aanwezig is
        if herstel_df is not None:
            dag_herstel = herstel_df[herstel_df["dag"] == etappe_nr]
            if not dag_herstel.empty:
                df_schoon["hrv_rmssd"] = dag_herstel["hrv_rmssd"].iloc[0]
                df_schoon["readiness_score"] = dag_herstel["readiness_score"].iloc[0]
            else:
                df_schoon["hrv_rmssd"] = np.nan
                df_schoon["readiness_score"] = np.nan
        else:
            df_schoon["hrv_rmssd"] = np.nan
            df_schoon["readiness_score"] = np.nan

        alle_etappes.append(df_schoon)

    master_df = pd.concat(alle_etappes, ignore_index=True)
    return master_df.sort_values(["etappe_nr", "tijd"]).reset_index(drop=True)