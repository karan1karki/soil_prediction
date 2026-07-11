import pandas as pd
df = pd.read_excel('/Crop_recommendation.xls')
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# scatter plot of Nitrogen vs Potassium
plt.figure(figsize=(10, 6))
sns.scatterplot(x='N', y='K', data=df, hue='label')
plt.title('Scatter Plot of Nitrogen vs Potassium')
plt.xlabel('Nitrogen (N)')
plt.ylabel('Potassium (K)')
plt.legend(title='Crop Type')
plt.show()

# pair plot of all features
sns.pairplot(df, hue='label')
plt.show()