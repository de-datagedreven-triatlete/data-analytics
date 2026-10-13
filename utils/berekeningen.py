"""Berekeningen voor de analyses van trainingen en wedstrijden.

Deze module bevat alleen berekeningen en geen grafieken. Alle functies gaan uit
van het vaste DataFrame-formaat dat `utils.inlezen.schoon_fit_data` oplevert
(kolommen `tijd`, `hartslag`, `hoogte`, `afstand_m`, `snelheid_kmh`, `cadans`, ...).
Standaardwaarden komen uit `utils.config.ANALYSE`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from utils.config import ANALYSE
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Algemene hulpfuncties
# ---------------------------------------------------------------------------
def gladstrijken_savgol(waarden, venster: int, polyorder: int) -> np.ndarray:
    """Strijkt een reeks glad met een Savitzky-Golay-filter.

    Het venster wordt automatisch oneven gemaakt en nooit groter dan de reeks.
    Is de reeks te kort voor het filter, dan komt de oorspronkelijke reeks terug.
    Ontbrekende waarden worden vooraf lineair aangevuld.
    """
    reeks = pd.Series(np.asarray(waarden, dtype=float)).interpolate().ffill().bfill()
    lengte = len(reeks)
    venster = int(venster) + (1 if int(venster) % 2 == 0 else 0)
    langste_oneven = lengte if lengte % 2 == 1 else lengte - 1
    venster = min(venster, langste_oneven)
    if venster <= polyorder or reeks.isna().any():
        return reeks.to_numpy()
    return savgol_filter(reeks.to_numpy(), window_length=venster, polyorder=polyorder)


def bereken_helling(
    df: pd.DataFrame,
    min_pct: float = ANALYSE.helling_min_pct,
    max_pct: float = ANALYSE.helling_max_pct,
) -> pd.Series:
    """Berekent de helling (%) tussen twee opeenvolgende metingen.

    Helling = hoogteverschil / afgelegde afstand x 100. Waar geen afstand is
    afgelegd (stilstand), is de helling onbepaald (NaN). Extreme waarden worden
    afgekapt tussen `min_pct` en `max_pct`, omdat die door meetruis ontstaan.
    """
    hoogteverschil = df["hoogte"].diff()
    afstandsverschil = df["afstand_m"].diff().where(lambda s: s > 0)
    return (hoogteverschil / afstandsverschil * 100.0).clip(min_pct, max_pct)


def bereken_beweegtijden(df: pd.DataFrame) -> dict:
    """Berekent de verstreken tijd en de bewogen tijd (in seconden).

    Vereist de kolom `is_stil` (zie `markeer_pauzes`). Alleen metingen met een
    hartslagwaarde tellen mee als bewogen tijd.
    """
    verstreken_s = (df["tijd"].iloc[-1] - df["tijd"].iloc[0]).total_seconds()
    tijdverschil_s = df["tijd"].diff().dt.total_seconds().fillna(0)
    bewogen_s = tijdverschil_s[~df["is_stil"] & df["hartslag"].notna()].sum()
    return {"verstreken_s": verstreken_s, "bewogen_s": float(bewogen_s)}


# ---------------------------------------------------------------------------
# Pauzes
# ---------------------------------------------------------------------------
def markeer_pauzes(df: pd.DataFrame, drempel_kmh: float, min_duur_s: float) -> pd.DataFrame:
    """Markeert metingen waarin de atleet stilstaat en aaneengesloten rustpauzes.

    Voegt toe:
        is_stil        Snelheid ligt onder `drempel_kmh`.
        pauze_groep    Volgnummer van de aaneengesloten reeks stilstaande/bewegende metingen.
        pauze_duur_s   Duur van de reeks (tijd tussen eerste en laatste meting).
        is_pauze       Stilstand die minstens `min_duur_s` duurt.
    """
    df = df.copy()
    df["is_stil"] = df["snelheid_kmh"] < drempel_kmh
    df["pauze_groep"] = (df["is_stil"] != df["is_stil"].shift()).cumsum()
    per_groep = df.groupby("pauze_groep")["tijd"]
    df["pauze_duur_s"] = (per_groep.transform("max") - per_groep.transform("min")).dt.total_seconds()
    df["is_pauze"] = df["is_stil"] & (df["pauze_duur_s"] >= min_duur_s)
    return df


# ---------------------------------------------------------------------------
# Klim-intensiteit (VAM)
# ---------------------------------------------------------------------------
def bereken_vam(
    df: pd.DataFrame,
    venster_s: int = ANALYSE.vam_venster_s,
    min_helling_pct: float = ANALYSE.vam_min_helling_pct,
) -> pd.DataFrame:
    """Berekent de VAM (Velocità Ascensionale Media, in m/u) over een rollend tijdvenster.

    Formule: VAM = (stijging in meters / tijd in seconden) x 3600.
    Alleen positieve hoogteverschillen tellen als stijging. Meetruis in de hoogte telt daardoor
    ook als stijging mee; gebruik bij een ruisige hoogtemeting eerst gladgestreken hoogtes.

    Voegt toe:
        tijdverschil_s       Tijd sinds de vorige meting.
        stijging_m           Stijging sinds de vorige meting (nooit negatief).
        totale_stijging_m    Opgetelde stijging sinds de start.
        helling_pct          Helling tussen twee metingen.
        helling_venster_pct  Gemiddelde helling over het tijdvenster.
        vam                  VAM over het tijdvenster (0 tot `ANALYSE.vam_max`).
        vam_klim             Zoals `vam`, maar alleen waar de gemiddelde helling minstens
                             `min_helling_pct` is. Op afdalingen en vlakke stukken is dit NaN.
        actieve_minuten      Verstreken tijd zonder tijdsprongen (pauzes) van meer dan 5 s.
        afstand_km           Afstand in kilometers.
    """
    df = df.sort_values("tijd").reset_index(drop=True)
    df["tijdverschil_s"] = df["tijd"].diff().dt.total_seconds()
    df["stijging_m"] = df["hoogte"].diff().clip(lower=0).fillna(0)
    df["totale_stijging_m"] = df["stijging_m"].cumsum()
    df["helling_pct"] = bereken_helling(df)

    op_tijd = df.set_index("tijd")
    venster = f"{int(venster_s)}s"
    stijging_venster = op_tijd["stijging_m"].rolling(venster).sum()
    tijd_venster = op_tijd["tijdverschil_s"].rolling(venster).sum()
    helling_venster = op_tijd["helling_pct"].rolling(venster).mean()

    vam = (stijging_venster / tijd_venster.where(tijd_venster > 0) * 3600.0).fillna(0.0)
    df["vam"] = vam.clip(0, ANALYSE.vam_max).to_numpy()
    df["helling_venster_pct"] = helling_venster.to_numpy()
    df["vam_klim"] = df["vam"].where(df["helling_venster_pct"] >= min_helling_pct)

    actief = df["tijdverschil_s"].clip(upper=ANALYSE.actieve_tijd_max_gat_s).fillna(0)
    df["actieve_minuten"] = actief.cumsum() / 60.0
    df["afstand_km"] = df["afstand_m"] / 1000.0
    return df


# ---------------------------------------------------------------------------
# Herstelhartslag
# ---------------------------------------------------------------------------
def maak_hartslagreeks_voor_weergave(
    df: pd.DataFrame,
    venster: int = ANALYSE.hrr_hartslag_glad,
    max_bevroren: int = ANALYSE.hrr_max_bevroren,
) -> pd.Series:
    """Maakt een nette hartslagreeks voor in een grafiek.

    Stappen:
        1. Hartslag tijdens stilstand weglaten (vereist kolom `is_stil`).
        2. Sensorstoringen weglaten: meer dan `max_bevroren` identieke waarden op rij.
        3. Voortschrijdend gemiddelde over `venster` metingen.
    """
    hartslag = df["hartslag"].copy()
    hartslag[df["is_stil"]] = np.nan

    gelijk_aan_vorige = df["hartslag"] == df["hartslag"].shift(1)
    reeks_lengte = gelijk_aan_vorige.groupby((~gelijk_aan_vorige).cumsum()).cumsum()
    hartslag[reeks_lengte > max_bevroren] = np.nan

    return hartslag.rolling(window=venster, min_periods=10).mean()


def bereken_herstelhartslag(
    df: pd.DataFrame,
    min_pauze_s: float = ANALYSE.hrr_min_pauze_s,
    min_daling_bpm: float = ANALYSE.hrr_min_daling_bpm,
    venster_s: float = ANALYSE.hrr_venster_s,
) -> pd.DataFrame:
    """Berekent per rustpauze de herstelhartslag (HRR).

    Herstelhartslag = hartslag bij het begin van de pauze minus de hartslag
    `venster_s` seconden later. Is de pauze korter dan het venster, dan geldt de
    laagste hartslag tijdens de pauze. Pauzes met minder dan `min_daling_bpm`
    herstel worden weggelaten.

    Vereist de kolommen `is_stil` en `pauze_groep` (zie `markeer_pauzes`) en `verstreken_minuten`.

    Returns:
        DataFrame met per pauze: `start_tijd`, `start_minuut`, `duur_min`,
        `hartslag_start` en `herstel_bpm`.
    """
    pauzes = []
    for _, groep in df[df["is_stil"]].groupby("pauze_groep"):
        duur_s = (groep["tijd"].iloc[-1] - groep["tijd"].iloc[0]).total_seconds()
        if duur_s < min_pauze_s:
            continue

        start_tijd = groep["tijd"].iloc[0]
        hartslag_start = groep["hartslag"].iloc[0]
        eerste_venster = groep[groep["tijd"] <= start_tijd + pd.Timedelta(seconds=venster_s)]

        if not eerste_venster.empty and duur_s >= venster_s:
            herstel = hartslag_start - eerste_venster["hartslag"].iloc[-1]
        else:
            herstel = hartslag_start - groep["hartslag"].min()

        if herstel >= min_daling_bpm:
            pauzes.append(
                {
                    "start_tijd": start_tijd,
                    "start_minuut": groep["verstreken_minuten"].iloc[0],
                    "duur_min": duur_s / 60.0,
                    "hartslag_start": hartslag_start,
                    "herstel_bpm": herstel,
                }
            )
    return pd.DataFrame(
        pauzes, columns=["start_tijd", "start_minuut", "duur_min", "hartslag_start", "herstel_bpm"]
    )


# ---------------------------------------------------------------------------
# Beweegratio
# ---------------------------------------------------------------------------
def bereken_beweegratio(
    raster: pd.DataFrame,
    drempel_kmh: float = ANALYSE.pauze_snelheid_kmh,
    min_pauze_s: float = ANALYSE.pauze_min_duur_s,
    venster: int = ANALYSE.savgol_venster_s,
    polyorder: int = ANALYSE.savgol_polyorder,
) -> tuple[pd.DataFrame, dict]:
    """Berekent welk deel van de tijd de atleet in beweging was.

    Args:
        raster: Resultaat van `utils.inlezen.maak_tijdraster` (één meting per seconde).
        drempel_kmh: Onder deze snelheid staat de atleet stil.
        min_pauze_s: Alleen stilstand vanaf deze duur telt als rustpauze.
        venster, polyorder: Instellingen van het Savitzky-Golay-filter voor de weergave.

    Returns:
        Het raster met de extra kolommen `is_pauze` en `snelheid_glad_kmh`, en een
        dictionary met `totale_tijd_s`, `stilstand_s`, `beweging_s` en `beweegratio_pct`.
    """
    df = markeer_pauzes(raster, drempel_kmh, min_pauze_s)
    df["snelheid_glad_kmh"] = np.clip(gladstrijken_savgol(df["snelheid_kmh"], venster, polyorder), 0, None)
    df.loc[df["is_pauze"], "snelheid_glad_kmh"] = 0.0

    totaal_s = (df["tijd"].max() - df["tijd"].min()).total_seconds()
    stilstand_s = float(df["is_pauze"].sum())
    beweging_s = totaal_s - stilstand_s
    resultaat = {
        "totale_tijd_s": totaal_s,
        "stilstand_s": stilstand_s,
        "beweging_s": beweging_s,
        "beweegratio_pct": beweging_s / totaal_s * 100.0 if totaal_s > 0 else 0.0,
    }
    return df, resultaat


# ---------------------------------------------------------------------------
# Cardiac drift
# ---------------------------------------------------------------------------
def bereken_cardiac_drift(
    raster: pd.DataFrame,
    drempel_kmh: float = ANALYSE.pauze_snelheid_kmh,
    min_pauze_s: float = ANALYSE.pauze_min_duur_s,
    venster: int = ANALYSE.savgol_venster_s,
    polyorder: int = ANALYSE.savgol_polyorder,
    hoogtecorrectie_factor: float = ANALYSE.hoogtecorrectie_factor,
) -> tuple[pd.DataFrame, dict]:
    """Berekent de cardiac drift: het oplopen van de hartslag bij gelijke inspanning.

    Methode:
        1. Rustpauzes verwijderen, zodat alleen actieve tijd overblijft.
        2. Snelheid corrigeren voor hoogtewinst. De stijgsnelheid (m/s) wordt omgerekend naar
           km/u en vermenigvuldigd met `hoogtecorrectie_factor`.
        3. De actieve tijd in twee gelijke helften splitsen.
        4. Per helft de efficiëntiefactor berekenen: gemiddelde hartslag / gemiddelde
           gecorrigeerde snelheid.
        5. Drift (%) = (EF helft 2 - EF helft 1) / EF helft 1 x 100.

    Let op: de hoogtecorrectie is een aanname (1 m stijging weegt als `hoogtecorrectie_factor` m
    horizontaal) en niet wetenschappelijk gevalideerd. De uitkomst is bedoeld voor vergelijking
    tussen etappes van dezelfde atleet, niet als absolute maat.

    Returns:
        De actieve data met extra kolommen `actieve_minuten`, `snelheid_glad_kmh`, `hartslag_glad`,
        `hoogte_glad` en `gecorrigeerde_snelheid_kmh`, en een dictionary met
        `gem_hartslag_helft1/2`, `ef_helft1/2` en `drift_pct`.
    """
    df = markeer_pauzes(raster, drempel_kmh, min_pauze_s)
    actief = df[~df["is_pauze"]].reset_index(drop=True)
    actief["actieve_minuten"] = np.arange(len(actief)) / 60.0

    actief["snelheid_glad_kmh"] = np.clip(gladstrijken_savgol(actief["snelheid_kmh"], venster, polyorder), 0, None)
    actief["hartslag_glad"] = gladstrijken_savgol(actief["hartslag"], venster, polyorder)
    actief["hoogte_glad"] = gladstrijken_savgol(actief["hoogte"], venster, polyorder)

    stijging_m_per_s = actief["hoogte"].diff().clip(lower=0).fillna(0)
    actief["gecorrigeerde_snelheid_kmh"] = actief["snelheid_kmh"] + stijging_m_per_s * 3.6 * hoogtecorrectie_factor

    midden = len(actief) // 2
    helften = (actief.iloc[:midden], actief.iloc[midden:])
    gem_hartslag = [helft["hartslag"].mean() for helft in helften]
    gem_snelheid = [helft["gecorrigeerde_snelheid_kmh"].mean() for helft in helften]
    ef = [hr / snelheid if snelheid > 0 else 0.0 for hr, snelheid in zip(gem_hartslag, gem_snelheid)]
    drift_pct = (ef[1] - ef[0]) / ef[0] * 100.0 if ef[0] > 0 else 0.0

    resultaat = {
        "gem_hartslag_helft1": gem_hartslag[0],
        "gem_hartslag_helft2": gem_hartslag[1],
        "ef_helft1": ef[0],
        "ef_helft2": ef[1],
        "drift_pct": drift_pct,
    }
    return actief, resultaat


# ---------------------------------------------------------------------------
# 2D- versus 3D-afstand
# ---------------------------------------------------------------------------
def haversine_afstand(breedte1, lengte1, breedte2, lengte2):
    """Berekent de afstand (m) over het aardoppervlak tussen twee punten (haversine-formule).

    De coördinaten zijn in graden. Werkt ook op hele kolommen (Series of arrays).
    """
    straal_aarde_m = 6_371_000
    phi1, phi2 = np.radians(breedte1), np.radians(breedte2)
    delta_phi = np.radians(breedte2 - breedte1)
    delta_lambda = np.radians(lengte2 - lengte1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    return straal_aarde_m * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bereken_3d_afstand(
    df: pd.DataFrame,
    hoogte_venster: int = ANALYSE.hoogte_3d_glad_venster,
    max_stap_m: float = ANALYSE.gps_max_stap_m,
) -> pd.DataFrame:
    """Berekent de afgelegde afstand op een vlakke kaart (2D) en over het echte terrein (3D).

    Per stap is de 3D-afstand sqrt(2D-afstand^2 + hoogteverschil^2).

    Meetruis in de hoogte telt bij elke stap als extra afstand. Daarom wordt de hoogte
    eerst gladgestreken met een gecentreerd voortschrijdend gemiddelde over `hoogte_venster`
    metingen (1 = niet gladstrijken). Stappen groter dan `max_stap_m` gelden als GPS-storing
    en tellen niet mee.

    Voegt toe: `hoogte_glad`, `stap_2d_m`, `stap_3d_m`, `afstand_2d_km`, `afstand_3d_km`,
    `extra_m` en `cumulatief_extra_m`.
    """
    if df["breedtegraad"].isna().all() or df["lengtegraad"].isna().all():
        raise ValueError("De data bevat geen GPS-posities; 2D- en 3D-afstand zijn niet te berekenen.")

    df = df.sort_values("tijd").reset_index(drop=True)
    hoogte = df["hoogte"].interpolate().ffill().bfill()
    if hoogte_venster > 1:
        hoogte = hoogte.rolling(window=hoogte_venster, center=True, min_periods=1).mean()
    df["hoogte_glad"] = hoogte

    stap_2d = haversine_afstand(
        df["breedtegraad"].shift(1), df["lengtegraad"].shift(1), df["breedtegraad"], df["lengtegraad"]
    ).fillna(0)
    stap_2d = stap_2d.where(stap_2d < max_stap_m, 0.0)

    hoogteverschil = hoogte.diff().fillna(0)
    stap_3d = np.sqrt(stap_2d**2 + hoogteverschil**2)

    df["stap_2d_m"] = stap_2d
    df["stap_3d_m"] = stap_3d
    df["afstand_2d_km"] = stap_2d.cumsum() / 1000.0
    df["afstand_3d_km"] = stap_3d.cumsum() / 1000.0
    df["extra_m"] = stap_3d - stap_2d
    df["cumulatief_extra_m"] = df["extra_m"].cumsum()
    return df


# ---------------------------------------------------------------------------
# Triatlon: fases en vergelijking van de runs
# ---------------------------------------------------------------------------
def detecteer_triatlonfasen(
    df: pd.DataFrame,
    fiets_snelheid_kmh: float = ANALYSE.fiets_snelheid_kmh,
    wissel_snelheid_kmh: float = ANALYSE.wissel_snelheid_kmh,
    t2_zoekvenster_min: float = ANALYSE.t2_zoekvenster_min,
    run2_snelheid_kmh: float = ANALYSE.run2_snelheid_kmh,
    reserve_min: float = ANALYSE.wisselduur_reserve_min,
) -> dict:
    """Bepaalt de fases van een run-bike-run op basis van de snelheid.

    Methode:
        * Fiets: alle metingen boven `fiets_snelheid_kmh`.
        * T1 begint bij het laatste aaneengesloten stilstandsblok (< `wissel_snelheid_kmh`) vóór de fiets.
        * T2 eindigt wanneer na de laatste stop binnen `t2_zoekvenster_min` na de fiets de snelheid
          weer boven `run2_snelheid_kmh` komt.
        * Kan een wissel niet worden gedetecteerd, dan wordt `reserve_min` minuten aangehouden.

    Vereist de kolommen `snelheid_kmh` en `verstreken_minuten`.

    Returns:
        Dictionary met `t1_start`, `fiets_start`, `fiets_eind`, `t2_eind` en `totaal` (alle in minuten)
        en `fases`: een lijst met per fase `naam`, `start` en `eind`.
    """
    snelheid = df["snelheid_kmh"]
    minuten = df["verstreken_minuten"]

    fiets = df[snelheid > fiets_snelheid_kmh]
    if fiets.empty:
        raise ValueError(f"Geen fietsfase gevonden: de snelheid komt nergens boven {fiets_snelheid_kmh} km/u.")
    fiets_start = fiets["verstreken_minuten"].min()
    fiets_eind = fiets["verstreken_minuten"].max()

    # T1: start van het laatste stilstandsblok voor de fiets
    stil_voor_fiets = df[(minuten < fiets_start) & (snelheid < wissel_snelheid_kmh)]
    if stil_voor_fiets.empty:
        t1_start = fiets_start - reserve_min
    else:
        blok = (stil_voor_fiets.index.to_series().diff() != 1).cumsum()
        laatste_blok = stil_voor_fiets[blok == blok.iloc[-1]]
        t1_start = laatste_blok["verstreken_minuten"].min()

    # T2: laatste stop vlak na de fiets, daarna het moment dat het lopen weer op gang komt
    stops_na_fiets = df[(minuten >= fiets_eind) & (minuten <= fiets_eind + t2_zoekvenster_min) & (snelheid < wissel_snelheid_kmh)]
    t2_eind = fiets_eind + reserve_min
    if not stops_na_fiets.empty:
        laatste_stop = stops_na_fiets["verstreken_minuten"].max()
        run2 = df[(minuten >= laatste_stop) & (snelheid > run2_snelheid_kmh)]
        if not run2.empty:
            t2_eind = run2["verstreken_minuten"].min()

    totaal = minuten.max()
    return {
        "t1_start": t1_start,
        "fiets_start": fiets_start,
        "fiets_eind": fiets_eind,
        "t2_eind": t2_eind,
        "totaal": totaal,
        "fases": [
            {"naam": "RUN 1", "start": 0.0, "eind": t1_start},
            {"naam": "T1", "start": t1_start, "eind": fiets_start},
            {"naam": "BIKE", "start": fiets_start, "eind": fiets_eind},
            {"naam": "T2", "start": fiets_eind, "eind": t2_eind},
            {"naam": "RUN 2", "start": t2_eind, "eind": totaal},
        ],
    }


def selecteer_runs_voor_vergelijking(
    df: pd.DataFrame, fiets_start: float, fiets_eind: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selecteert de metingen van run 1 en run 2 voor een eerlijke vergelijking.

    Alleen metingen binnen een vergelijkbaar tempo- en hartslagbereik blijven over
    (zie `ANALYSE.run_*`), zodat opwarmen, wandelen en uitlopen de vergelijking niet vertekenen.

    Returns:
        Twee DataFrames (run 1, run 2) met de extra kolommen `snelheid_glad_kmh` en `hartslag_glad`.
    """
    df = df.copy()
    venster, minimum = ANALYSE.run_glad_venster, ANALYSE.run_glad_min_metingen
    df["snelheid_glad_kmh"] = df["snelheid_kmh"].rolling(window=venster, min_periods=minimum).mean()
    df["hartslag_glad"] = df["hartslag"].rolling(window=venster, min_periods=minimum).mean()

    in_bereik = (
        df["snelheid_glad_kmh"].between(ANALYSE.run_snelheid_min_kmh, ANALYSE.run_snelheid_max_kmh)
        & (df["hartslag_glad"] > ANALYSE.run_hartslag_min)
    )
    kolommen = ["snelheid_glad_kmh", "hartslag_glad"]
    run1 = df[(df["verstreken_minuten"] < fiets_start) & in_bereik].dropna(subset=kolommen)
    run2 = df[(df["verstreken_minuten"] > fiets_eind) & in_bereik].dropna(subset=kolommen)
    return run1, run2


# ---------------------------------------------------------------------------
# Helling, cadans en hartslag op de berg
# ---------------------------------------------------------------------------
def analyseer_helling_cadans(
    df: pd.DataFrame, min_helling_pct: float = ANALYSE.klim_min_helling_pct
) -> tuple[pd.DataFrame, dict]:
    """Onderzoekt hoe de cadans (stappen per minuut) verandert met de helling.

    Alleen klimmen vanaf `min_helling_pct` met een geldige cadans (zie `ANALYSE.cadans_*`)
    tellen mee. Door deze punten wordt een rechte lijn (lineaire regressie) getrokken.

    Returns:
        De gebruikte klimpunten (met kolom `helling_pct`) en een dictionary met
        `hellingscoefficient` (spm per %), `intercept`, `verandering_per_10_pct` (spm per +10 %),
        `gemiddelde_cadans`, `max_helling_pct` en `aantal_punten`.
    """
    werk = df.copy()
    werk["helling_pct"] = bereken_helling(werk)

    geldige_cadans = werk["cadans"].between(ANALYSE.cadans_min_spm, ANALYSE.cadans_max_spm, inclusive="neither")
    klim = werk[(werk["helling_pct"] >= min_helling_pct) & geldige_cadans].dropna(subset=["helling_pct", "cadans"])
    if len(klim) < 2:
        raise ValueError("Te weinig klimpunten met een geldige cadans voor een regressie.")

    coefficient, intercept = np.polyfit(klim["helling_pct"], klim["cadans"], 1)
    resultaat = {
        "hellingscoefficient": coefficient,
        "intercept": intercept,
        "verandering_per_10_pct": coefficient * 10,
        "gemiddelde_cadans": klim["cadans"].mean(),
        "max_helling_pct": klim["helling_pct"].max(),
        "aantal_punten": len(klim),
    }
    return klim, resultaat


def bereken_hoogtemeters(df: pd.DataFrame) -> dict:
    """Telt de stijging en daling (m) op uit de hoogtemetingen.

    Dit is een eigen berekening uit ruwe metingen en wijkt daardoor iets af van de totalen
    die het horloge zelf berekent (zie `utils.inlezen.lees_sessiegegevens`).
    """
    verschil = df["hoogte"].diff()
    return {"stijging_m": float(verschil[verschil > 0].sum()), "daling_m": float(-verschil[verschil < 0].sum())}


def bereken_hellingsklassen(
    df: pd.DataFrame,
    onder_pct: int = ANALYSE.klasse_min_pct,
    boven_pct: int = ANALYSE.klasse_max_pct,
    breedte_pct: int = ANALYSE.klasse_breedte_pct,
    min_aantal: int = ANALYSE.klasse_min_aantal,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Berekent de gemiddelde hartslag per hellingsklasse.

    De hellingen worden in klassen van `breedte_pct` procent ingedeeld. Klassen met minder
    dan `min_aantal` metingen worden weggelaten, omdat het gemiddelde dan onbetrouwbaar is.

    Returns:
        De metingen met kolom `helling_pct`, en per klasse `midden`, `gemiddelde`,
        `standaarddeviatie` en `aantal`.
    """
    werk = df.copy()
    werk["helling_pct"] = bereken_helling(werk)
    klassen = pd.cut(werk["helling_pct"], bins=np.arange(onder_pct, boven_pct, breedte_pct))

    statistiek = (
        werk.groupby(klassen, observed=True)["hartslag"]
        .agg(gemiddelde="mean", standaarddeviatie="std", aantal="count")
        .reset_index(names="klasse")
    )
    statistiek["midden"] = statistiek["klasse"].apply(lambda interval: interval.mid)
    statistiek = statistiek[statistiek["aantal"] >= min_aantal].dropna(subset=["gemiddelde"]).reset_index(drop=True)
    return werk, statistiek


# ---------------------------------------------------------------------------
# Cadanstrend en baseline over meerdere etappes
# ---------------------------------------------------------------------------
def bereken_klimcadans(df: pd.DataFrame, min_helling_pct: float = ANALYSE.baseline_klim_helling_pct) -> float:
    """Berekent de gemiddelde cadans (stappen per minuut) op stukken steiler dan `min_helling_pct`.

    Metingen met cadans 0 (stilstaan) tellen niet mee.
    """
    cadans = df["cadans"].replace(0, np.nan)
    helling = bereken_helling(df)
    return float(cadans[helling >= min_helling_pct].mean())


def bereken_klimcadans_per_bestand(
    fit_map: str | Path,
    van_datum: str,
    tot_datum: str,
    naamfilter: str = "stubaier_hohenweg",
    cadans_factor: float = ANALYSE.wandel_cadans_factor,
    min_helling_pct: float = ANALYSE.baseline_klim_helling_pct,
) -> pd.DataFrame:
    """Berekent de klimcadans voor elk .fit-bestand in een map.

    Args:
        fit_map: Map met .fit-bestanden. Bestandsnamen beginnen met de datum (JJJJMMDD).
        van_datum, tot_datum: Periode (JJJJMMDD, grenzen inbegrepen) die als baseline telt.
        naamfilter: Alleen bestanden waarvan de naam deze tekst bevat worden verwerkt.
        cadans_factor: Vermenigvuldigingsfactor voor de cadans (2 = stappen per minuut).
        min_helling_pct: Minimale helling voor een klim.

    Returns:
        DataFrame met per bestand `bestand`, `datum`, `klimcadans_spm` en `in_baseline`.
        Bestanden die niet te verwerken zijn, worden overgeslagen en gelogd.
    """
    resultaten = []
    for pad in sorted(Path(fit_map).glob("*.fit")):
        if naamfilter not in pad.name:
            continue
        datum = pad.name.split("_")[0]
        try:
            df = schoon_fit_data(lees_fit_bestand(pad), cadans_factor=cadans_factor)
            klimcadans = bereken_klimcadans(df, min_helling_pct)
        except Exception as fout:  # Eén kapot bestand mag de hele reeks niet stoppen
            logger.warning("Overgeslagen: %s (%s)", pad.name, fout)
            continue
        resultaten.append(
            {
                "bestand": pad.name,
                "datum": datum,
                "klimcadans_spm": klimcadans,
                "in_baseline": van_datum <= datum <= tot_datum,
            }
        )
    return pd.DataFrame(resultaten, columns=["bestand", "datum", "klimcadans_spm", "in_baseline"])


def bereken_baseline(samenvatting: pd.DataFrame, kolom: str = "klimcadans_spm") -> float:
    """Berekent de baseline: het gemiddelde van de bestanden met `in_baseline` is waar."""
    baseline = samenvatting.loc[samenvatting["in_baseline"], kolom]
    if baseline.empty:
        raise ValueError("Geen enkel bestand valt binnen de baseline-periode.")
    return float(baseline.mean())


def maak_terreinprofiel(df: pd.DataFrame, rooster_m: float = ANALYSE.terrein_rooster_m) -> pd.DataFrame:
    """Zet hoogte en cadans om naar een regelmatig afstandsraster.

    De metingen liggen niet op gelijke afstanden (stilstaan geeft veel metingen op dezelfde plek).
    Door te interpoleren naar een raster van `rooster_m` meter ontstaat een gelijkmatige reeks
    waarop helling en cadans eerlijk te vergelijken zijn. Cadans 0 (stilstaan) wordt genegeerd.

    Returns:
        DataFrame met `afstand_km`, `hoogte`, `cadans` en `helling_pct`.
    """
    werk = df.dropna(subset=["afstand_m"]).sort_values("tijd")
    afstand_km = np.maximum.accumulate(werk["afstand_m"].to_numpy()) / 1000.0
    hoogte = werk["hoogte"].interpolate().ffill().bfill().to_numpy()
    cadans = werk["cadans"].replace(0, np.nan).interpolate().ffill().bfill().to_numpy()

    raster_km = np.arange(afstand_km[0], afstand_km[-1], rooster_m / 1000.0)
    if len(raster_km) < 2:
        raise ValueError("De afgelegde afstand is te kort voor een terreinprofiel.")

    profiel = pd.DataFrame(
        {
            "afstand_km": raster_km,
            "hoogte": np.interp(raster_km, afstand_km, hoogte),
            "cadans": np.interp(raster_km, afstand_km, cadans),
        }
    )
    profiel["helling_pct"] = np.gradient(profiel["hoogte"]) / np.gradient(profiel["afstand_km"] * 1000.0) * 100.0
    return profiel


def filter_terrein(
    profiel: pd.DataFrame, terrein: str = "klim", grens_pct: float = ANALYSE.terrein_grens_pct
) -> pd.DataFrame:
    """Selecteert het terreintype uit een terreinprofiel.

    Args:
        terrein: 'klim' (helling >= grens), 'afdaling' (helling <= -grens) of 'vlak' (daartussen).
    """
    if terrein == "klim":
        masker = profiel["helling_pct"] >= grens_pct
    elif terrein == "afdaling":
        masker = profiel["helling_pct"] <= -grens_pct
    elif terrein == "vlak":
        masker = profiel["helling_pct"].between(-grens_pct, grens_pct)
    else:
        raise ValueError("Onbekend terreintype; kies 'klim', 'afdaling' of 'vlak'.")
    return profiel[masker].copy()


def bereken_cadanstrend(
    gefilterd: pd.DataFrame,
    venster: int = ANALYSE.cadanstrend_venster,
    polyorder: int = ANALYSE.cadanstrend_polyorder,
) -> pd.DataFrame:
    """Voegt de kolom `cadans_glad` toe: de cadans gladgestreken met een Savitzky-Golay-filter."""
    gefilterd = gefilterd.copy()
    gefilterd["cadans_glad"] = gladstrijken_savgol(gefilterd["cadans"], venster, polyorder) if len(gefilterd) > 1 else gefilterd["cadans"]
    return gefilterd

# ---------------------------------------------------------------------------
# Steilste segment
# ---------------------------------------------------------------------------
def bereken_steilste_segment(
    df: pd.DataFrame,
    doel_afstand_m: float = 500.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Zoekt automatisch het steilste segment van een bepaalde lengte in een beklimming.
   
    Methode:
        1. Vindt het hoogste punt in de activiteit om de hoofdklim te isoleren.
        2. Berekent de helling per meetpunt.
        3. Schuift een venster van `doel_afstand_m` over de klim om de sectie met
           de maximale hoogtewinst te vinden.
           
    Returns:
        Vier elementen: data van de klim, data van de rest van de etappe, data van
        het steilste segment, en een dictionary met vergelijkende statistieken.
    """
    werk = df.copy()
    werk["helling_pct"] = bereken_helling(werk)
    werk["afstand_km"] = werk["afstand_m"] / 1000.0

    # 1. Splitsen in klim (tot de top) en rest (na de top)
    piek_idx = werk["hoogte"].idxmax()
    df_klim = werk.loc[:piek_idx].copy()
    df_rest = werk.loc[piek_idx:].copy()

    # 2. Steilste segment zoeken door er een venster overheen te schuiven
    beste_segment = df_klim.iloc[:0]
    max_stijging = -1

    for i in range(len(df_klim)):
        start_afst = df_klim["afstand_m"].iloc[i]
        eind_afst_doel = start_afst + doel_afstand_m
       
        # Sub-dataframe voor dit venster
        sub_df = df_klim[(df_klim["afstand_m"] >= start_afst) & (df_klim["afstand_m"] <= eind_afst_doel)]
       
        if not sub_df.empty:
            stijging = sub_df["hoogte"].iloc[-1] - sub_df["hoogte"].iloc[0]
            if stijging > max_stijging:
                max_stijging = stijging
                beste_segment = sub_df

    # 3. Lokale hulpfunctie voor statistieken
    def bereken_stats(sub_df):
        if len(sub_df) < 2:
            return {"duur_min": 0.0, "vam": 0, "hartslag": 0, "gem_helling_pct": 0.0}
           
        duur_s = (sub_df["tijd"].iloc[-1] - sub_df["tijd"].iloc[0]).total_seconds()
        stijging = sub_df["hoogte"].iloc[-1] - sub_df["hoogte"].iloc[0]
       
        gem_helling = sub_df.loc[sub_df["helling_pct"] > 0, "helling_pct"].mean()
       
        return {
            "duur_min": duur_s / 60.0 if duur_s > 0 else 1.0,
            "vam": int(stijging / (duur_s / 3600.0)) if duur_s > 0 else 0,
            "hartslag": int(sub_df["hartslag"].mean()) if sub_df["hartslag"].notna().any() else 0,
            "gem_helling_pct": gem_helling if pd.notna(gem_helling) else 0.0
        }

    statistieken = {
        "klim": bereken_stats(df_klim),
        "segment": bereken_stats(beste_segment)
    }

    return df_klim, df_rest, beste_segment, statistieken
