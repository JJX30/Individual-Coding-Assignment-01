import pandas as pd
import matplotlib.pyplot as plt
import display as ds

data1 = pd.read_csv("articleInfo.csv")
data2 = pd.read_csv("authorInfo.csv")

output1 = pd.merge(data1, data2, on="Article No.", how="outer").fillna(0)

output1.to_csv("merged.csv")

ds.regression()
# # Displays both graphs one after the other
ds.display()

# # # Creates SVG of countries in directory
# ds.countryPlot()
