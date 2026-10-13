# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 6: 2D- versus 3D-afstand (de hoogtefactor)
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 6 (17 september 2026)
# MAGIC
# MAGIC **Vraag:** Hoeveel verder loop ik werkelijk in het steile terrein dan een platte kaart aangeeft?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De 2D-afstand tussen twee GPS-punten volgt uit de haversine-formule (afstand over het aardoppervlak).
# MAGIC 2. De hoogte wordt eerst gladgestreken, omdat meetruis anders als extra afstand meetelt.
# MAGIC 3. De 3D-afstand per stap is de wortel uit (2D-afstand in het kwadraat + hoogteverschil in het kwadraat).
# MAGIC 4. Stappen groter dan 200 m gelden als GPS-storing en tellen niet mee.
# MAGIC 5. Het verschil tussen 3D en 2D zijn de extra steilte-meters.
# MAGIC
# MAGIC **Resultaat:** een afbeelding in de map `output/`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installatie van de benodigde pakketten
# MAGIC
# MAGIC De pakketten staan in `requirements.txt`. Deze cel hoeft alleen de eerste keer na het opstarten van het cluster te draaien.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from utils.berekeningen import bereken_3d_afstand
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_hoogtefactor
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 6
FIT_BESTAND = PADEN.fit_map / "20260917_stubaier_hohenweg_etappe_6.fit"
HOOGTE_VENSTER = ANALYSE.hoogte_3d_glad_venster 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data inlezen

# COMMAND ----------

ruwe_data = lees_fit_bestand(FIT_BESTAND)
df = schoon_fit_data(ruwe_data, cadans_factor=ANALYSE.wandel_cadans_factor)
df = bereken_verstreken_tijd(df)

print(f"{len(df)} metingen ingelezen: {df['tijd'].min():%d-%m-%Y %H:%M} t/m {df['tijd'].max():%H:%M}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berekening: 2D- en 3D-afstand

# COMMAND ----------

df_3d = bereken_3d_afstand(df, hoogte_venster=HOOGTE_VENSTER)

print(f"2D-afstand: {df_3d['afstand_2d_km'].iloc[-1]:.2f} km")
print(f"3D-afstand: {df_3d['afstand_3d_km'].iloc[-1]:.2f} km")
print(f"Extra:      {df_3d['cumulatief_extra_m'].iloc[-1]:.0f} m")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_hoogtefactor(df_3d, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260917_elevation_factor", formaat="liggend")