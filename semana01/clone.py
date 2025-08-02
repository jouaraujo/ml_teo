#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# %%
df = pd.read_parquet("data/dados_clones.parquet")
df.head()

# %%
model = tree.DecisionTreeClassifier()

# %%
df["General Jedi encarregado"].unique()

#%%
df = df.replace({
        "Tipo 1": 1, "Tipo 2": 2, "Tipo 3": 3, "Tipo 4": 4, "Tipo 5":5,
})

df

# %%
features = ["Massa(em kilos)",
              "Estatura(cm)", "Distância Ombro a ombro", "Tamanho do crânio",
              "Tamanho dos pés", "Tempo de existência(em meses)"]

target = "Status "

X = df[features]
y = df[target]

# %%
X

# %%
y

# %%
model.fit(X, y)

# %%
plt.figure(dpi=400)

tree.plot_tree(model,
               max_depth=3,
               feature_names=features,
               class_names=model.classes_,
               filled=True);

#%%
df.groupby("Status ")[["Massa(em kilos)", "Estatura(cm)"]].mean()

