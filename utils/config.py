"""Centrale instellingen voor alle analyses.

Alle drempelwaarden, paden en kleuren staan hier, zodat de berekeningen en
grafieken herbruikbaar blijven en een aanpassing op één plek volstaat.
De waarden zijn standaardwaarden: een notebook kan ze per analyse overschrijven
door ze als argument aan een functie mee te geven.
"""
from dataclasses import dataclass
from pathlib import Path

# De projectmap is de map die boven 'utils' ligt. Zo werken de paden ongeacht
# vanuit welke map een notebook wordt uitgevoerd.
PROJECTMAP = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AtleetConfig:
    """Persoonlijke fysiologische gegevens van de atleet."""

    max_hartslag: int = 187        # Maximale hartslag (slagen per minuut)
    drempel_hartslag: int = 173    # Hartslag op de anaerobe drempel (slagen per minuut)
    ftp: int = 197                 # Functional Threshold Power (watt)


@dataclass(frozen=True)
class PadConfig:
    """Locaties van de invoer- en uitvoermappen."""

    fit_map: Path = PROJECTMAP / "fit_bestanden"   # Ruwe .fit-bestanden (niet in Git)
    data_map: Path = PROJECTMAP / "data"           # Kleine aanvullende invoer (csv)
    uitvoer_map: Path = PROJECTMAP / "output"      # Gegenereerde afbeeldingen


@dataclass(frozen=True)
class AnalyseConfig:
    """Standaard drempelwaarden en vensterlengtes voor de analyses."""

    # --- Algemeen ---
    max_gat_s: int = 15                  # Tijdgat (s) waarboven twee metingen niet als aaneengesloten gelden
    wandel_cadans_factor: float = 2.0    # Garmin registreert bij lopen/wandelen de cadans per been; x2 = stappen per minuut
    helling_min_pct: float = -30.0       # Hellingen daaronder worden als meetruis afgekapt
    helling_max_pct: float = 45.0        # Hellingen daarboven worden als meetruis afgekapt
    actieve_tijd_max_gat_s: int = 5      # Tijdsprongen groter dan dit tellen niet mee als actieve tijd

    # --- Triatlonfasen (run-bike-run) ---
    fiets_snelheid_kmh: float = 20.0     # Boven deze snelheid is het fietsen
    wissel_snelheid_kmh: float = 3.0     # Onder deze snelheid staat de atleet stil in de wisselzone
    t2_zoekvenster_min: float = 5.0      # Zoekvenster (min) na de fiets voor de tweede wissel
    run2_snelheid_kmh: float = 8.5       # Boven deze snelheid is de tweede run begonnen
    wisselduur_reserve_min: float = 2.0  # Aangenomen wisselduur (min) als de wissel niet te detecteren is

    # --- Vergelijking run 1 en run 2 ---
    run_snelheid_min_kmh: float = 8.0
    run_snelheid_max_kmh: float = 15.0
    run_hartslag_min: float = 140.0      # Lagere hartslagen horen bij opwarmen of uitlopen
    run_glad_venster: int = 20           # Aantal metingen voor het voortschrijdend gemiddelde
    run_glad_min_metingen: int = 5

    # --- Klim-intensiteit (VAM) ---
    vam_venster_s: int = 300             # Tijdvenster (s) voor de VAM-berekening
    vam_max: float = 3000.0              # Hogere waarden zijn meetruis
    vam_min_helling_pct: float = 1.0     # Alleen klimmen vanaf deze helling tonen
    hoogte_glad_venster: int = 15        # Aantal metingen voor het gladstrijken van de hoogte

    # --- Helling versus cadans ---
    klim_min_helling_pct: float = 1.0
    cadans_min_spm: float = 40.0         # Geldige wandel-/loopcadans (stappen per minuut)
    cadans_max_spm: float = 220.0

    # --- Herstelhartslag ---
    hrr_stilstand_kmh: float = 1.62      # Onder deze snelheid (0,45 m/s) staat de atleet stil
    hrr_min_pauze_s: int = 180           # Minimale pauzeduur (s)
    hrr_min_daling_bpm: float = 5.0      # Kleinere herstelwaarden worden niet getoond
    hrr_venster_s: int = 60              # Herstel wordt gemeten over de eerste 60 s van de pauze
    hrr_hartslag_glad: int = 60          # Aantal metingen voor het gladstrijken van de hartslag
    hrr_max_bevroren: int = 30           # Meer dan zoveel identieke hartslagen op rij = sensorstoring

    # --- Pauzes, beweegratio en cardiac drift ---
    pauze_snelheid_kmh: float = 0.5      # Onder deze snelheid staat de atleet stil
    pauze_min_duur_s: int = 150          # Alleen aaneengesloten stilstand vanaf deze duur is een rustpauze
    savgol_venster_s: int = 301          # Venster (s) voor het Savitzky-Golay-filter
    savgol_polyorder: int = 2
    hoogtecorrectie_factor: float = 10.0  # Aanname: 1 m stijging weegt als 10 m horizontaal (zie bereken_cardiac_drift)

    # --- 2D- versus 3D-afstand ---
    gps_max_stap_m: float = 200.0        # Grotere sprongen tussen twee metingen zijn GPS-storingen
    hoogte_3d_glad_venster: int = 5      # Hoogte gladstrijken voordat de 3D-afstand wordt berekend

    # --- Cadanstrend per terreintype ---
    terrein_grens_pct: float = 5.0       # Klimmen vanaf +5 %, dalen vanaf -5 %
    terrein_rooster_m: float = 10.0      # Afstandsraster waarop de data wordt geïnterpoleerd
    cadanstrend_venster: int = 75
    cadanstrend_polyorder: int = 3
    baseline_klim_helling_pct: float = 5.0

    # --- Hartslag per hellingsklasse ---
    klasse_min_pct: int = -35
    klasse_max_pct: int = 40
    klasse_breedte_pct: int = 5
    klasse_min_aantal: int = 10          # Klassen met minder metingen worden weggelaten


# Standaardinstanties die in de notebooks worden gebruikt
ATLEET = AtleetConfig()
PADEN = PadConfig()
ANALYSE = AnalyseConfig()

# Uniform kleurenpalet voor alle grafieken
KLEUREN = {
    "achtergrond": "#000000",
    "kaart": "#1E222D",
    "hartslag": "#FF2A6D",       # Neon koraal
    "snelheid": "#00F5FF",       # IJscyaan
    "fiets": "#FFD700",          # Goud
    "gedempt": "#8A99AD",        # Leisteengrijs
    "hoogte": "#8A99AD",         # Hoogteprofiel op de achtergrond
    "raster": "#2A2A2A",
    "tekst": "#FFFFFF",
    "rust": "#EF4444",           # Rustpauzes
    "blok": "#111827",           # Achtergrond van statistiekenblokken
    "blokrand": "#374151",
    "goed": "#00C853",
    "waarschuwing": "#FFA500",
}

# Afmetingen (breedte, hoogte in inches) per exportformaat
EXPORTFORMATEN = {
    "staand": (10.8, 13.5),      # 4:5 (Instagram-post)
    "vierkant": (10.8, 10.8),    # 1:1
    "verhaal": (10.8, 19.2),     # 9:16 (story/reel)
    "liggend": (13.5, 7.59),     # 16:9 (breedbeeld)
}
