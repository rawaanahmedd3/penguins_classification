import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

missing_values = ["na", "N/A", np.nan]
df = pd.read_csv("penguins.csv", na_values=missing_values, low_memory=False)

for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

labelencoder = LabelEncoder()
df['gender'] = labelencoder.fit_transform(df['gender'])

first = df.iloc[:50, :]
second = df.iloc[50:100, :]
third = df.iloc[100:150, :]

plt.scatter(x=first['bill_length_mm'], y=first['bill_depth_mm'], color='blue')
plt.scatter(x=second['bill_length_mm'], y=second['bill_depth_mm'], color='green')
plt.scatter(x=third['bill_length_mm'], y=third['bill_depth_mm'], color='orange')
plt.xlabel("bill length (mm)")
plt.ylabel("bill depth (mm)")
plt.show()

plt.scatter(x=first['bill_length_mm'], y=first['flipper_length_mm'], color='blue')
plt.scatter(x=second['bill_length_mm'], y=second['flipper_length_mm'], color='green')
plt.scatter(x=third['bill_length_mm'], y=third['flipper_length_mm'], color='orange')
plt.xlabel("bill length (mm)")
plt.ylabel("flipper length (mm)")
plt.show()

plt.scatter(x=first['bill_length_mm'], y=first['gender'], color='blue')
plt.scatter(x=second['bill_length_mm'], y=second['gender'], color='green')
plt.scatter(x=third['bill_length_mm'], y=third['gender'], color='orange')
plt.xlabel("bill length (mm)")
plt.ylabel("gender")
plt.show()

plt.scatter(x=first['bill_length_mm'], y=first['body_mass_g'], color='blue')
plt.scatter(x=second['bill_length_mm'], y=second['body_mass_g'], color='green')
plt.scatter(x=third['bill_length_mm'], y=third['body_mass_g'], color='orange')
plt.xlabel("bill length (mm)")
plt.ylabel("body mass (g)")
plt.show()

plt.scatter(x=first['bill_depth_mm'], y=first['flipper_length_mm'], color='blue')
plt.scatter(x=second['bill_depth_mm'], y=second['flipper_length_mm'], color='green')
plt.scatter(x=third['bill_depth_mm'], y=third['flipper_length_mm'], color='orange')
plt.xlabel("bill depth (mm)")
plt.ylabel("flipper length (mm)")
plt.show()

plt.scatter(x=first['bill_depth_mm'], y=first['gender'], color='blue')
plt.scatter(x=second['bill_depth_mm'], y=second['gender'], color='green')
plt.scatter(x=third['bill_depth_mm'], y=third['gender'], color='orange')
plt.xlabel("bill depth (mm)")
plt.ylabel("gender")
plt.show()

plt.scatter(x=first['bill_depth_mm'], y=first['body_mass_g'], color='blue')
plt.scatter(x=second['bill_depth_mm'], y=second['body_mass_g'], color='green')
plt.scatter(x=third['bill_depth_mm'], y=third['body_mass_g'], color='orange')
plt.xlabel("bill depth (mm)")
plt.ylabel("body mass (g)")
plt.show()

plt.scatter(x=first['flipper_length_mm'], y=first['gender'], color='blue')
plt.scatter(x=second['flipper_length_mm'], y=second['gender'], color='green')
plt.scatter(x=third['flipper_length_mm'], y=third['gender'], color='orange')
plt.xlabel("flipper length (mm)")
plt.ylabel("gender")
plt.show()

plt.scatter(x=first['flipper_length_mm'], y=first['body_mass_g'], color='blue')
plt.scatter(x=second['flipper_length_mm'], y=second['body_mass_g'], color='green')
plt.scatter(x=third['flipper_length_mm'], y=third['body_mass_g'], color='orange')
plt.xlabel("flipper length (mm)")
plt.ylabel("body mass (g)")
plt.show()

plt.scatter(x=first['gender'], y=first['body_mass_g'], color='blue')
plt.scatter(x=second['gender'], y=second['body_mass_g'], color='green')
plt.scatter(x=third['gender'], y=third['body_mass_g'], color='orange')
plt.xlabel("gender")
plt.ylabel("body mass (g)")
plt.show()
