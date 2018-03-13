import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np


df = pd.read_csv("C:\\Users\\shachar\\Downloads\\train.csv")
fig = plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar" , alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age,alpha=0.1)
plt.title("Age wrt survived")


plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar" , alpha=0.5)
plt.title("Pclass")




plt.show()
