#%%
import pandas as pd

# %%
df = pd.read_csv("data/Dados Comunidade (respostas) - dados.csv")
df.head()

# %%
df.shape

# %%
df = df.replace({"Sim": 1, "Não": 0})
df.head()

# %%
num_vars = [
    "Curte games?",
    "Curte futebol?",
    "Curte livros?",
    "Curte jogos de tabuleiro?",
    "Curte jogos de fórmula 1?",
    "Curte jogos de MMA?",
    "Idade",
]

dummy_vars = [
    "Como conheceu o Téo Me Why?", "Quantos cursos acompanhou do Téo Me Why?", 
    "Estado que mora atualmente", "Área de Formação", "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]


df_analise = pd.get_dummies(df[dummy_vars]).astype(int)

df_analise[num_vars] = df[num_vars].copy()

# %%
df_analise

# %%
df_analise["pessoa feliz?"] = df["Você se considera uma pessoa feliz?"].copy()
df_analise

# %%
from sklearn import tree

# %%
features = df_analise.columns[:-1].to_list()
features

#%%
X = df_analise[features]
y = df_analise["pessoa feliz?"]

arvore = tree.DecisionTreeClassifier(random_state=42,
                                     min_samples_leaf=5,
                                     )

arvore.fit(X, y)

# %%
arvore_predict = arvore.predict(X)
arvore_predict

# %%
df_predict = df_analise[["pessoa feliz?"]]
df_predict["predict_arvore"] = arvore_predict
df_predict

# %%
df_predict.sample(10)

# %%
# Acurácia
(df_predict["pessoa feliz?"] == df_predict["predict_arvore"]).mean()

# %%
pd.crosstab(df_predict["pessoa feliz?"], df_predict["predict_arvore"])

# %%
(df_predict['pessoa feliz?'] == 1).sum()

# %%
