# ANALYSE FACTORIELLE DISCRIMINANTE

#########################################################################################################

# Variables à définir
    # Import et traitement des données
path = 'C:/Users/Dwimo/Documents/05 DATA/00 GITHUB/Training/Human activity/Data/train.csv' # emplacement des données
sep = ',' # séparateur de colonnes
index = None # nom ou numero de colonne
drop_col = ['subject', 'Activity'] # nom des colonnes à ignorer pour l'ACP
nb_components = 2

# Affichage des figures (1 pour oui, 0 pour non)
scree_plot = 1 # % de variance portée par les dimensions
corr_circle = 0 # cercle des corrélations
nb_dimensions = 2 # nombre de dimensions à afficher (2, 4, 6)
data_plot = 1 # graphique des individus
var_color = 'Activity' # choix d'une variable pour colorer les individus

#########################################################################################################

# import des librairies
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
raw_data = pd.read_csv(path, sep=sep, index_col=index)

# Préparation des données pour l'ACP
data = raw_data.drop(drop_col, axis=1)
data = data.fillna(raw_data.mean()) # remplacement des valeurs manquantes par la moyenne de la variable
X = data.values
y = raw_data['Activity']

# Centrage et réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Recherche des composantes principales
lda = LinearDiscriminantAnalysis(solver='svd', n_components=nb_components)
lda.fit(X_scaled, y)

# Projection des individus sur les axes factoriels
X_projected = lda.transform(X_scaled)

# GRAPHIQUES
plt.style.use('seaborn-whitegrid')
fontsize_axes = 12
fontsize_ticks = 10
fontsize_title = 14

def display_scree_plot(lda):
    plt.figure(figsize=(8,5))
    scree = lda.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree, color="#34738C")

    # legends
    plt.xlabel("Dimensions", fontsize=fontsize_axes)
    plt.xticks(fontsize=fontsize_ticks)
    plt.ylabel("Percentage of explained variances", fontsize=fontsize_axes)
    plt.yticks(fontsize=fontsize_ticks)
    plt.title("Scree plot", fontsize=fontsize_title, loc='left')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.show()

if scree_plot == 1:
    display_scree_plot(lda)

def display_correlation_circle(lda, dim=(1,2)): 
    pcs = lda.components_
    
    # figure
    plt.figure(figsize=(6,5))
    plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[dim[0]-1,:], pcs[dim[1]-1,:], 
                   np.hypot(pcs[dim[0]-1,:], pcs[dim[1]-1,:]),
                   angles='xy', scale_units='xy', scale=1, 
                   cmap="viridis", width=0.003)

    # vectors' names
    for i, (x, y) in enumerate(zip(pcs[dim[0]-1, :], pcs[dim[1]-1, :])):
        ha='left'
        va = 'bottom'
        if x < 0:
            ha='right'
        if y < 0:
            va='top'
        plt.text(x, y, raw_data.columns[i], fontsize=fontsize_ticks,
                 ha=ha, va=va) # vectors' names

    # circle
    circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='black', linewidth=0.5)
    plt.gca().add_artist(circle)

    # (0,0) horizontal and vertical lines 
    plt.plot([-1, 1], [0, 0], color='black', ls='--', linewidth=0.5)
    plt.plot([0, 0], [-1, 1], color='black', ls='--', linewidth=0.5)

    # graph limits
    plt.xlim([-1.05, 1.05])
    plt.ylim([-1.05, 1.05])

    # legends
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label('Norm', labelpad=5)
    plt.xlabel('Dim{} ({}%)'.format(dim[0], round(100*lda.explained_variance_ratio_[dim[0]-1],1)), fontsize=fontsize_axes)
    plt.xticks(fontsize=fontsize_ticks)
    plt.ylabel('Dim{} ({}%)'.format(dim[1], round(100*lda.explained_variance_ratio_[dim[1]-1],1)), fontsize=fontsize_axes)
    plt.yticks(fontsize=fontsize_ticks)
    plt.title("Correlation circle", fontsize=fontsize_title, loc='left')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    plt.show()

if corr_circle == 1:
    display_correlation_circle(lda, dim=(1,2))
    if nb_dimensions > 2:
        display_correlation_circle(lda, dim=(3,4))
        if nb_dimensions > 4:
            display_correlation_circle(lda, dim=(5,6))

def display_projected_data(lda, dim=(1,2)):
    # figure
    plt.figure(figsize=(8,5))

    if var_color != '':
        sns.scatterplot(X_projected[:,dim[0]-1], X_projected[:,dim[1]-1], 
                    hue=raw_data[var_color], cmap="viridis", s=12)
    else:
        plt.scatter(X_projected[:,dim[0]-1], X_projected[:,dim[1]-1], 
                    c="#34738C", s=12)

    # graph limits
    xmin, xmax, ymin, ymax = min(X_projected[:,dim[0]-1]), max(X_projected[:,dim[0]-1]), min(X_projected[:,dim[1]-1]), max(X_projected[:,dim[1]-1])
    plt.xlim([xmin-0.5, xmax+0.5])
    plt.ylim([ymin-0.5, ymax+0.5])

    # legends
    plt.xlabel('Dim{} ({}%)'.format(dim[0], round(100*lda.explained_variance_ratio_[dim[0]-1],1)), fontsize=fontsize_axes)
    plt.xticks(fontsize=fontsize_ticks)
    plt.ylabel('Dim{} ({}%)'.format(dim[1], round(100*lda.explained_variance_ratio_[dim[1]-1],1)), fontsize=fontsize_axes)
    plt.yticks(fontsize=fontsize_ticks)
    plt.title("Projected data", fontsize=fontsize_title, loc='left')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    plt.show()

if data_plot == 1:
    display_projected_data(lda, dim=(1,2))
    if nb_dimensions > 2:
        display_projected_data(lda, dim=(3,4))
        if nb_dimensions > 4:
            display_projected_data(lda, dim=(5,6))

# Prédiction

X_test = pd.read_csv('Data/test.csv', sep=sep, index_col=index)
y_test = X_test['Activity']
X_test = X_test.drop(drop_col, axis=1)

X_test_scaled = std_scale.transform(X_test)
X_test = lda.transform(X_test)

