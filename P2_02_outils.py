""" Librairie personnelle pour visualisation/graph multiple...
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Outils divers
# Version : 0.0.1 - CRE LR 06/02/2021s
# ====================================================================
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import missingno
from IPython.display import display
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.3'

# --------------------------------------------------------------------
# -- TYPES DES VARIABLES
# --------------------------------------------------------------------
def get_types_variables(df_work, types, type_par_var, graph):
    """ Permet un aperçu du type des variables
    Parameters
    ----------
    @param IN : df_work : dataframe, obligatoire
                types : Si True lance dtypes, obligatoire
                type_par_var : Si True affiche tableau des types de
                               chaque variable, obligatoire
                graph : Si True affiche pieplot de répartition des types
    @param OUT :None.
    """

    if types :
        # 1. Type des variables
        print("-------------------------------------------------------------")
        print("Type de variable pour chacune des variables\n")
        display(df_work.dtypes)

    if type_par_var :
        # 2. Compter les types de variables
        #print("Répartition des types de variable\n")
        values = df_work.dtypes.value_counts()
        nb_tot = values.sum()
        percentage = round((100 * values / nb_tot),2)
        table = pd.concat([values, percentage], axis=1)
        table.columns = ['Nombre par type de variable'
                         , '% des types de variable']
        display(table[table['Nombre par type de variable'] != 0]
                .sort_values('% des types de variable', ascending = False)
                .style.background_gradient('Greens'))

    if graph :
        # 3. Schéma des types de variable
        #print("\n----------------------------------------------------------")
        #print("Répartition schématique des types de variable \n")
        # Répartition des types de variables
        df_work.dtypes.value_counts().plot.pie()
        plt.ylabel('')
        plt.show()


# ---------------------------------------------------------------------------
# -- VALEURS MANQUANTES
# ---------------------------------------------------------------------------

# Afficher des informations sur les valeurs manquantes
def get_missing_values(df_work, pourcentage, heatmap):
    """Indicateurs sur les variables manquantes
       @param in : df_work dataframe obligatoire
                   pourcentage : boolean si True affiche le nombre heatmap
                   heatmap : boolean si True affiche la heatmap
       @param out : none
    """

    # 1. Nombre de valeurs manquantes totales
    nb_nan_tot=df_work.isna().sum().sum()
    nb_donnees_tot=np.product(df_work.shape)
    pourc_nan_tot=round((nb_nan_tot/nb_donnees_tot)*100,2)
    print(f'Valeurs manquantes :{nb_nan_tot} NaN pour {nb_donnees_tot} données ({pourc_nan_tot} %)')

    if pourcentage:
        print("-------------------------------------------------------------")
        print("Nombre et pourcentage de valeurs manquantes par variable\n")
        # 2. Visualisation du nombre et du pourcentage de valeurs manquantes par variable
        values = df_work.isnull().sum()
        percentage = 100 * values / len(df_work)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = ['Nombres de valeurs manquantes'
                         ,'% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0]
                .sort_values('% de valeurs manquantes', ascending = False)
                .style.background_gradient('Greens'))

    if heatmap:
        print("-------------------------------------------------------------")
        print("Heatmap de visualisation des valeurs manquantes")
        # 3. Heatmap de visualisation des valeurs manquantes
        plt.figure(figsize=(20, 10))
        sns.heatmap(df_work.isna(), cbar=False)
        plt.show()

# ---------------------------------------------------------------------------
# -- EDA DES TIME SERIES
# ---------------------------------------------------------------------------
def time_series_plot(df_work):
    """Given dataframe, generate times series plot of numeric data by daily,
       monthly and yearly frequency"""
    print("\nTo check time series of numeric data  by daily, monthly and yearly frequency")
    if len(df_work.select_dtypes(include='datetime64').columns)>0:
        for col in df_work.select_dtypes(include='datetime64').columns:
            for plotting in ['D', 'M', 'Y']:
                if plotting=='D':
                    print("Plotting daily data")
                elif plotting=='M':
                    print("Plotting monthly data")
                else:
                    print("Plotting yearly data")
                for col_num in df_work.select_dtypes(include=np.number).columns:
                    __ = df_work.copy()
                    __ = __.set_index(col)
                    transp = __.resample(plotting).sum()
                    axes = transp[[col_num]].plot()
                    axes.set_ylim(bottom=0)
                    axes.get_yaxis().set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                    plt.show()

# --------------------------------------------------------------------------
# -- EDA DES VARIABLES QUANTITATIVES
# --------------------------------------------------------------------------

# Génère EDA pour les variables quantitatives du dataframe transmis
def numeric_eda(df_work, hue=None):
    """Génère EDA pour les variables quantitatives du dataframe transmis
       @param in : df_work dataframe obligatoire
                   hue non obigatoire
       @param out : none
    """
    print("----------------------------------------------------")
    print("\nEDA variables quantitatives : \nDistribution des variables quantitatives\n")
    print(df_work.describe().T)
    columns = df_work.select_dtypes(include=np.number).columns
    figure = plt.figure(figsize=(20, 10))
    figure.add_subplot(1, len(columns), 1)
    for index, col in enumerate(columns):
        if index > 0:
            figure.add_subplot(1, len(columns), index + 1)
        sns.boxplot(y=col, data=df_work, boxprops={'facecolor': 'None'})
    figure.tight_layout()
    plt.show()

    if len(df_work.select_dtypes(include='category').columns) > 0:
        for col_num in df_work.select_dtypes(include=np.number).columns:
            for col in df_work.select_dtypes(include='category').columns:
                fig = sns.catplot(x=col, y=col_num, kind='violin', data=df_work, height=5, aspect=2)
                fig.set_xticklabels(rotation=90)
                plt.show()

    # Affiche le pairwise joint distributions
    print("\nAffiche pairplot des variables quantitatives")
    if hue is None:
        sns.pairplot(df_work.select_dtypes(include=np.number))
    else:
        sns.pairplot(df_work.select_dtypes(include=np.number).join(df_work[[hue]]), hue=hue)
    plt.show()

# --------------------------------------------------------------------------
# -- EDA DES VARIABLES QUALITATIVES
# --------------------------------------------------------------------------

# Top 5 des modalités uniques par variable qualitative
def top5(df_work):
    """Affiche le top 5 des modalités uniques par variables qualitatives
       @param in : df_work dataframe obligatoire
       @param out : none
    """
    print("----------------------------------------------------")
    columns = df_work.select_dtypes(include=['object', 'category']).columns
    for col in columns:
        print("Top 5 des modalités uniques de : " + col)
        print(df_work[col].value_counts().reset_index()
              .rename(columns={"index": col, col: "Count"})[
              :min(5, len(df_work[col].value_counts()))])
        print(" ")


# Génère EDA pour les variables qualitatives du dataframe transmis
def categorical_eda(df_work, hue=None):
    """Génère EDA pour les variables qualitatives du dataframe transmis
       @param in : df_work dataframe obligatoire
                   hue non obigatoire
       @param out : none
    """
    print("----------------------------------------------------")
    print("\nEDA variables qualitatives : \nDistribution des variables qualitatives")
    print(df_work.select_dtypes(include=['object', 'category']).nunique())
    top5(df_work)
    # Affiche count distribution des variables qualitatives
    for col in df_work.select_dtypes(include='category').columns:
        fig = sns.catplot(x=col, kind="count", data=df_work, hue=hue)
        fig.set_xticklabels(rotation=90)
        plt.show()

# ---------------------------------------------------------------------------
# -- EDA DE TOUTES LES VARIABLES : QUANTITATIVES, QUALITATIVES
# ---------------------------------------------------------------------------
def eda(df_work):
    """Génère l'analyse exploratoire du dataframe transmis pour toutes les variables"""

    print("----------------------------------------------------")

    # Controle que le paramètre transmis est un dataframe pandas
    # if type(df_work) != pd.core.frame.DataFrame:
    if isinstance(df_work, pd.core.frame.DataFrame):
        raise TypeError("Seul un dataframe pandas est autorisé en entrée")

    # Remplace les données avec vide ou espace par NaN
    df_work = df_work.replace(r'^\s*$', np.nan, regex=True)

    print("----------------------------------------------------")
    print("3 premières lignes du jeu de données:")
    print(df_work.head(3))

    print("----------------------------------------------------")
    print("\nEDA des variables: \n (1) Total du nombre de données \n  \
          (2) Types ds colonnes \n (3) Any null values\n")
    print(df_work.info())

    # Affichage des valeurs manquantes
    if df_work.isnull().any(axis=None):
        print("----------------------------------------------------")
        print("\nPrévisualisation des données avec valeurs manquantes:")
        print(df_work[df_work.isnull().any(axis=1)].head(3))
        missingno.matrix(df_work)
        plt.show()

    get_missing_values(df_work, True, True)

    print("----------------------------------------------------")
    # Statitstique du nombre de données dupliquées
    if len(df_work[df_work.duplicated()]) > 0:
        print("\n***Nombre de données dupliquées : ", len(df_work[df_work.duplicated()]))
        print(df_work[df_work.duplicated(keep=False)].sort_values(by=list(df_work.columns)).head())
    else:
        print("\nAucune donnée dupliquée trouvée")

    # EDA des variables qualitatives
    print("----------------------------------------------------")
    print('-- EDA DES VARIABLES QUALITATIVES')
    categorical_eda(df_work)

    # EDA des variables quantitatives
    print("----------------------------------------------------")
    print('-- EDA DES VARIABLES QUANTITATIVES')
    numeric_eda(df_work)

    # Affiche les Plot time series plot des variables quantitatives
    time_series_plot(df_work)

# ---------------------------------------------------------------------------
# -- Graph densité pour 1 ou plusieurs colonne d'un dataframe
# ---------------------------------------------------------------------------
def plot_graph(df_work):
    """Graph densité pour 1 ou plusieurs colonne d'un dataframe
       @param in : df_work dataframe obligatoire
       @param out : none
    """

    plt.figure(figsize=(10,5))
    axes = plt.axes()

    label_patches = []
    colors=['Crimson','SeaGreen','Sienna','DodgerBlue','Purple']

    i=0
    for col in df_work.columns:
        label=col
        sns.kdeplot(df_work[col])
        label_patch = mpatches.Patch(
            color=colors[i],
            label=label)
        label_patches.append(label_patch)
        i+=1
    plt.xlabel('')
    plt.legend(handles=label_patches, bbox_to_anchor=(1.05, 1)
               , loc=2, borderaxespad=0., facecolor='white')
    plt.grid(False)
    axes.set_facecolor('white')

    plt.show()
