# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 9: impact van de helling op de hartslag
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 9 (20 september 2026)
# MAGIC
# MAGIC **Vraag:** Welke hartslag hoort bij welke helling, en hoe verhoudt dalen zich tot klimmen?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De helling tussen twee metingen wordt berekend en ingedeeld in klassen van 5%. Stilstand heeft geen helling en telt niet mee.
# MAGIC 2. Per klasse worden de gemiddelde hartslag en de standaarddeviatie berekend. Klassen met minder dan 10 metingen worden weggelaten.
# MAGIC 3. De totale stijging, daling en gemiddelde hartslag komen uit de samenvatting die het horloge in het FIT-bestand opslaat. Ontbreken die, dan worden ze uit de metingen berekend.
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

from utils.berekeningen import bereken_hellingsklassen, bereken_hoogtemeters
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_helling_hartslag
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, lees_sessiegegevens, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 9
FIT_BESTAND = PADEN.fit_map / "20260920_stubaier_hohenweg_etappe_9.fit"

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
# MAGIC ## Berekening: hellingsklassen en totalen

# COMMAND ----------

df_helling, statistiek = bereken_hellingsklassen(df)

# Totalen: eerst de waarden van het horloge, anders zelf berekend uit de metingen
sessie = lees_sessiegegevens(FIT_BESTAND)
eigen_meters = bereken_hoogtemeters(df)
stijging_m = sessie.get("total_ascent", eigen_meters["stijging_m"])
daling_m = sessie.get("total_descent", eigen_meters["daling_m"])
gem_hartslag = sessie.get("avg_heart_rate", df["hartslag"].mean())

bron = "horloge (session)" if "total_ascent" in sessie else "eigen berekening"
print(f"Totalen uit: {bron}")
print(f"Stijging {stijging_m:.0f} m, daling {daling_m:.0f} m, gem. hartslag {gem_hartslag:.0f} bpm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_helling_hartslag(df_helling, statistiek, gem_hartslag, stijging_m, daling_m, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260920_descent_impact", formaat="liggend")