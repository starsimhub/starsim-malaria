"""
Analyze monthly data
"""

import io

import sciris as sc
import matplotlib.pyplot as plt
import pandas as pd

csv_data = '''
month,dielmo_1990,ndiop_1993,sugungum_1970,matsari_1970,rafin_marke_1970,laye_2007,dapelogo_2007,namawala_1991
1,10.4,0.39,2,4,1,1,1,43.8
2,13,0.19,1,1,1,1,1,68.5
3,6,0.77,0.1,0.1,0.5,1,1,27.4
4,2.6,0,0.1,0.1,1,1,1,46.6
5,4,0,0.1,1,1,1,1,49.4
6,6,0,5,2.5,2,7,3,24.7
7,35,6.4,9.625,7.75,3.875,8,27,13.7
8,21,2.2,10.375,11.625,7.75,9,70,11
9,28,4.7,20,56.25,15,5,130,11
10,15,3.9,23.25,27.125,3.875,4,57,2.74
11,10.4,0.87,4.125,12,1,6,29,13.7
12,8.4,0.58,4,6,1,1,1,16.5
'''

# Parse into dataframe (StringIO: read_csv expects a path or file-like, not a raw str)
df = sc.dataframe.read_csv(io.StringIO(csv_data.strip()))
df = df.set_index('month')

df = df.div(df.max(axis=0), axis=1)
eir_mean = df.mean(axis=1)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for col in df.columns:
    ax.plot(df.index, df[col], label=col, alpha=0.5)
ax.plot(df.index, eir_mean, color='black', linewidth=3, label='mean')
ax.legend()
ax.set_xlabel('Month')
ax.set_ylabel('EIR')
ax.set_title('Monthly EIR Data')
plt.show()
