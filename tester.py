# Load your data
from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors

df_history = pd.read_csv("spotify2023data.csv", encoding="utf-16")
print(df_history.columns)  # This will print all column names
