# Databricks notebook source
# MAGIC %md
# MAGIC ### Jointure entre les 6 dataframe

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE NBOSport (
# MAGIC SELECT i.*
# MAGIC       , t.TV_DURATION_CULTURE_L3M_SECONDES
# MAGIC       , t.TV_DURATION_CULTURE_L3M_DAYS
# MAGIC       , t.TV_RANK_CULTURE_L3M
# MAGIC       , t.TV_DURATION_CYCLING_L3M_SECONDES
# MAGIC       , t.TV_DURATION_CYCLING_L3M_DAYS
# MAGIC       , t.TV_RANK_CYCLING_L3M
# MAGIC       , t.TV_DURATION_DOCUMENTARIES_L3M_SECONDES
# MAGIC       , t.TV_DURATION_DOCUMENTARIES_L3M_DAYS
# MAGIC       , t.TV_RANK_DOCUMENTARIES_L3M
# MAGIC       , t.TV_DURATION_FOOTBALL_L3M_SECONDES
# MAGIC       , t.TV_DURATION_FOOTBALL_L3M_DAYS
# MAGIC       , t.TV_RANK_FOOTBALL_L3M
# MAGIC       , t.TV_DURATION_MOVIES_L3M_SECONDES
# MAGIC       , t.TV_DURATION_MOVIES_L3M_DAYS
# MAGIC       , t.TV_RANK_MOVIES_L3M
# MAGIC       , m.AMOUNT_OUT_OF_BUNDLE_L3M
# MAGIC       , m.NB_DAYS_OUT_OF_BUNDLE_L3M
# MAGIC       , s.AVG_CUST_AGE
# MAGIC       , s.AVG_NB_DAYS_INTERNET_LINE
# MAGIC       , s.AVG_NB_DAYS_TV_LINE
# MAGIC       , s.AVG_NB_MOBILE_SUBS
# MAGIC       , j.AMOUNT_OF_CONTACT_ADMINISTRATIVE_ISSUES_L3M
# MAGIC       , j.AMOUNT_OF_CONTACT_COMPLAINTS_L3M
# MAGIC       , j.JOURNEY_DURATION_ADMINISTRATIVE_ISSUES_L3M_DAYS
# MAGIC       , j.JOURNEY_DURATION_COMPLAINTS_L3M_DAYS
# MAGIC       , j.JOURNEY_DURATION_TECHNICAL_ISSUES_L3M_DAYS
# MAGIC       , j.TOTAL_NUMBER_OF_CONTACTS_L3M
# MAGIC       , j.TOTAL_NUMBER_OF_JOURNEYS_L3M
# MAGIC       , r.EVENT_TYPE
# MAGIC FROM nboTVAggregateExclM_1 AS T
# MAGIC FULL JOIN nboInternetAggregateExclM_1 AS I
# MAGIC   ON T.CUST_NUM = I.CUST_NUM AND T.YEAR = I.YEAR AND T.MONTH = I.MONTH
# MAGIC
# MAGIC LEFT JOIN nboMobileAggregateExclM_1 AS M
# MAGIC   ON COALESCE(T.CUST_NUM, I.CUST_NUM) = M.CUST_NUM
# MAGIC  AND COALESCE(T.YEAR, I.YEAR) = M.YEAR
# MAGIC  AND COALESCE(T.MONTH, I.MONTH) = M.MONTH
# MAGIC
# MAGIC LEFT JOIN nboSocioDemoAggregateExclM_1 AS S
# MAGIC   ON COALESCE(T.CUST_NUM, I.CUST_NUM) = S.CUST_NUM
# MAGIC  AND COALESCE(T.YEAR, I.YEAR) = S.YEAR
# MAGIC  AND COALESCE(T.MONTH, I.MONTH) = S.MONTH
# MAGIC
# MAGIC LEFT JOIN nboJourneysAggregateExclM_1 AS J
# MAGIC   ON COALESCE(T.CUST_NUM, I.CUST_NUM) = J.CUST_NUM
# MAGIC  AND COALESCE(T.YEAR, I.YEAR) = J.YEAR
# MAGIC  AND COALESCE(T.MONTH, I.MONTH) = J.MONTH
# MAGIC
# MAGIC LEFT JOIN nboRequestsAggregateExclM_1 AS R
# MAGIC   ON COALESCE(T.CUST_NUM, I.CUST_NUM) = R.CUST_NUM
# MAGIC  AND COALESCE(T.YEAR, I.YEAR) = R.YEAR
# MAGIC  AND COALESCE(T.MONTH, I.MONTH) = R.MONTH
# MAGIC );

# COMMAND ----------

df = spark.table("NBOSport")

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Création des variables à expliquer

# COMMAND ----------

from pyspark.sql import functions as F

# Sélection des événements liés au cinéma
events = df.filter(F.col("EVENT_TYPE") == "SPORT REQUEST") \
    .select("CUST_NUM", "YEAR", "MONTH", "EVENT_TYPE")

# Décalage des événements d’un mois en arrière (pour servir de cible au mois précédent)
events_shifted = events.withColumn("MONTH", F.col("MONTH") - 1)
events_shifted = events_shifted.withColumn(
    "YEAR",
    F.when(F.col("MONTH") == 0, F.col("YEAR") - 1).otherwise(F.col("YEAR"))
)
events_shifted = events_shifted.withColumn(
    "MONTH",
    F.when(F.col("MONTH") == 0, F.lit(12)).otherwise(F.col("MONTH"))
)

# Création de la colonne cible
events_shifted = events_shifted.withColumn(
    "TARGET_SPORT",
    F.when(F.col("EVENT_TYPE") == "SPORT REQUEST", 1)
)

# Agrégation (au cas où plusieurs événements par client/mois)
targets = events_shifted.groupBy("CUST_NUM", "YEAR", "MONTH").agg(
    F.max("TARGET_SPORT").alias("TARGET_SPORT")
)

# Jointure avec le dataset principal
df = df.join(targets, on=["CUST_NUM", "YEAR", "MONTH"], how="left") \
    .fillna({"TARGET_SPORT": 0})

# On peut supprimer EVENT_TYPE car il n’est plus utile
df = df.drop("EVENT_TYPE")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistiques

# COMMAND ----------

summary_df = df.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
summary_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pourcentage de valeurs nulles

# COMMAND ----------

from pyspark.sql.functions import col, sum as Fsum

row_count = df.count()

null_pct = df.select([
    (Fsum(col(c).isNull().cast("int")) / row_count).alias(c)
    for c in df.columns
])
null_pct.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remplacement des valeurs nulles par des '-1'

# COMMAND ----------

df = df.fillna(-1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matrice de corrélation

# COMMAND ----------

#from pyspark.sql.types import IntegerType, FloatType, DoubleType, LongType, ShortType, ByteType
#import pandas as pd

# Définir les types numériques
#numeric_types = (IntegerType, FloatType, DoubleType, LongType, ShortType, ByteType)

# Garder uniquement les colonnes numériques
#numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, numeric_types)]

# Calculer les corrélations paire à paire
#corr_dict = {}
#for c1 in numeric_cols:
#    corr_dict[c1] = []
#    for c2 in numeric_cols:
#        corr = df.stat.corr(c1, c2)
#        corr_dict[c1].append(round(corr, 2))

# Convertir en DataFrame Pandas
#correlation_df = pd.DataFrame(corr_dict, index=numeric_cols)
#correlation_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split train/test

# COMMAND ----------

# Split du DataFrame avec une seed pour reproductibilité
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

#print(f"Nombre de lignes train : {train_df.count()}")
#print(f"Nombre de lignes test  : {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Création modèle de machine learning (prédiction TARGET_SPORT)

# COMMAND ----------

pip install xgboost

# COMMAND ----------

pip install imblearn

# COMMAND ----------

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# 1. Préparation des données
train_pd = train_df.toPandas()
test_pd  = test_df.toPandas()

target_col = "TARGET_SPORT"
excluded = [target_col, "YEAR", "CUST_NUM", "MONTH"]
feature_cols = train_pd.select_dtypes(include="number").columns.drop(excluded)

X_train = train_pd[feature_cols].astype(float)
y_train = train_pd[target_col].astype(int)
X_test  = test_pd[feature_cols].astype(float)
y_test  = test_pd[target_col].astype(int)

# 2. Rééquilibrage avec SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("Distribution après SMOTE :", np.bincount(y_train_res))

# 3. Entraînement XGBoost
model_sport = XGBClassifier(
    tree_method="hist",
    eval_metric="logloss",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_sport.fit(X_train_res, y_train_res)

# 4. Wrapper pour forcer le modèle à renvoyer les probabilités
class XGBProbaWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

wrapped_model = XGBProbaWrapper(model_sport)

# 5. Signature : sortie attendue = probas entre 0 et 1
signature = infer_signature(X_train_res, model_sport.predict_proba(X_train_res)[:, 1])

# 6. Log du modèle via pyfunc (et non plus mlflow.xgboost)
mlflow.pyfunc.log_model(
    artifact_path="model_sport",
    python_model=wrapped_model,
    registered_model_name="sport_model",
    signature=signature
)

# 7. Évaluation locale pour contrôle
y_proba = model_sport.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)
print(f"\nAUC = {auc_score:.4f}")

threshold = 0.16
y_pred = (y_proba >= threshold).astype(int)

print(f"\n===== Metrics (threshold={threshold}) =====")
print("Confusion matrix :\n", confusion_matrix(y_test, y_pred))
print("\nClassification report :\n", classification_report(y_test, y_pred))

# 8. Résultat final (aligné version Spark)
df_final = test_pd.copy()
df_final["y_proba"] = y_proba
df_final["PREDICTION_SPORT"] = y_pred
df_final["THRESHOLD_USED"] = threshold

# Réorganisation des colonnes pour lisibilité
ordered_cols = (
    ["CUST_NUM", "YEAR", "MONTH", target_col, "PREDICTION_SPORT", "y_proba", "THRESHOLD_USED"]
    + [col for col in feature_cols if col in df_final.columns]
)
df_final = df_final[ordered_cols]

# COMMAND ----------

import mlflow

client = mlflow.tracking.MlflowClient()

# Nom complet du modèle dans Unity Catalog
model_full_name = "amavla_dev_central_adbw.default.sport_model"

# Récupérer toutes les versions
versions = client.search_model_versions(f"name='{model_full_name}'")
latest_version = max(int(v.version) for v in versions)

# Mettre à jour l’alias "last" pour pointer vers cette version
client.set_registered_model_alias(
    name=model_full_name,
    alias="last",
    version=latest_version
)

print(f"Alias 'last' mis à jour vers la version {latest_version}")

# COMMAND ----------

# ----------------------------------------
# Analyse par quantiles
# ----------------------------------------

# Dataframe pour analyse : probas + y_test
df_eval = pd.DataFrame({
    "y_true": y_test,
    "y_proba": y_proba
})

# Trier les individus du plus haut score au plus faible
df_eval = df_eval.sort_values(by="y_proba", ascending=False).reset_index(drop=True)

# Ajouter le quantile (par découpage en 20 parts égales → 5% chacune)
df_eval["quantile"] = pd.qcut(df_eval.index, 20, labels=np.arange(1, 21))

# Agrégation par quantile
agg = df_eval.groupby("quantile").agg(
    segment_target=("y_true", "sum"),
    segment_total=("y_true", "count")
).reset_index()

# Calcul des colonnes cumulées
agg["running_target"] = agg["segment_target"].cumsum()
agg["running_total"] = agg["segment_total"].cumsum()

# Baseline : % de positifs dans tout le jeu de test
baseline_precision = df_eval["y_true"].mean()

# Calcul des performances cumulées
agg["running_precision"] = agg["running_target"] / agg["running_total"]
agg["running_recall"] = agg["running_target"] / df_eval["y_true"].sum()
agg["running_lift"] = agg["running_precision"] / baseline_precision
agg["running_f1score"] = 2 * (agg["running_precision"] * agg["running_recall"]) / (
    agg["running_precision"] + agg["running_recall"] + 1e-9
)

# Renommer le quantile de 1 (top) à 20 (bottom)
agg["quantile"] = agg["quantile"].astype(int)

# Trier du top 5% au bottom 5%
agg = agg.sort_values("quantile")

# Affichage des résultats par quantile
pd.set_option("display.max_rows", 25)
print("\nAnalyse par quantiles (5% chacun) :\n")
agg.display()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --------------------------------------------------------
# Visualisation de l'importance des variables du modèle
# --------------------------------------------------------

# Récupérer les importances (gain) des features
booster = model_sport.get_booster()
importance_dict = booster.get_score(importance_type='gain')  # 'weight', 'cover', or 'gain'

# Convertir en DataFrame et compléter avec les variables absentes
importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain']).reset_index()
importance_df.columns = ['feature', 'gain']

# Ajouter les features manquantes avec gain = 0 (non utilisées)
all_features = list(feature_cols)
missing_features = list(set(all_features) - set(importance_df['feature']))
for f in missing_features:
    importance_df = pd.concat([importance_df, pd.DataFrame({'feature': [f], 'gain': [0.0]})], ignore_index=True)

# Centrer les barres autour de 0 pour un effet visuel neutre
importance_df["direction"] = np.where(importance_df["gain"] >= 0, "positive", "negative")
importance_df = importance_df.sort_values("gain", ascending=False)

# Tracer avec seaborn
plt.figure(figsize=(10, len(importance_df) * 0.4))
sns.barplot(
    data=importance_df,
    y="feature", x="gain",
    hue="direction",
    dodge=False,
    palette={"positive": "green", "negative": "red"}
)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.title("Importance des variables (type = gain)")
plt.xlabel("Gain moyen apporté à la prédiction")
plt.ylabel("Variable")
plt.legend().remove()
plt.tight_layout()
plt.show()

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Liste des seuils à tester
thresholds = np.arange(0.01, 0.91, 0.05)

results = []

for thr in thresholds:
    y_pred = (y_proba >= thr).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        "threshold": round(thr, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    })

# Mettre en DataFrame
df_thresholds = pd.DataFrame(results)

# Affichage complet
pd.set_option("display.max_rows", None)
df_thresholds.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conversion du dataframe en delta table + test

# COMMAND ----------

# Conversion Pandas -> Spark pour sauvegarde Delta
df_final = spark.createDataFrame(df_final)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS nboTrainSport;

# COMMAND ----------

# Sauver le dataframe au format delta table natif de Databricks
df_final.write.format("delta").saveAsTable("nboTrainSport")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test ouverture en SQL de la delta table
# MAGIC SELECT * 
# MAGIC FROM nboTrainSport;

# COMMAND ----------

# MAGIC %sql
# MAGIC select sum(TARGET_SPORT), sum(PREDICTION_SPORT)
# MAGIC from nboTrainSport
# MAGIC where TARGET_SPORT = 1;
