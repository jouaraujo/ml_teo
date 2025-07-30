#%%
import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

# %%
from sklearn import linear_model

X = df[["cerveja"]] # X maísculo porque é uma matriz diferente de y que é um vetor
y = df["nota"]

# %%
# Isso aqui é o aprendizado de máquina
reg = linear_model.LinearRegression()

reg.fit(X, y)

# %%
a, b = reg.intercept_, reg.coef_[0]
print(a, b)

# %%
predict = reg.predict(X.drop_duplicates())

# %%
import matplotlib.pyplot as plt

plt.plot(X["cerveja"], y, "o")
plt.grid()
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()["cerveja"], predict)

plt.legend(["Observado", f"y = {a:.3f} + {b:.3f} x"])
