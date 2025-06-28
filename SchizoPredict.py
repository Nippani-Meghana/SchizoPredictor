#Predict Schizophrenia from Simulated Brain Scores :A Logistic Regression Exploration (Fake dataset)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
np.random.seed(42)
n_samples = 500

age = np.random.randint(18, 66, size=n_samples)
gender = np.random.choice(['Male', 'Female'], size=n_samples)
working_memory_score = np.random.randint(0, 31, size=n_samples)
reaction_time = np.random.randint(300, 1501, size=n_samples)
panss_positive = np.random.randint(7, 50, size=n_samples)
panss_negative = np.random.randint(7, 50, size=n_samples)
data_gender =[]
for g in gender:
    if g == 'Male':
        data_gender.append(0)
    else:
        data_gender.append(1)
diagnosis = []
for pos, neg, wm in zip(panss_positive, panss_negative, working_memory_score):
    if pos > 20 and neg > 20 and wm < 15:
        diagnosis.append(1)
    else:
        diagnosis.append(0)

data = pd.DataFrame({
    'Age': age,
    'Gender': data_gender,
    'WorkingMemoryScore': working_memory_score,
    'ReactionTime': reaction_time,
    'PANSS_Positive': panss_positive,
    'PANSS_Negative': panss_negative,
    'Diagnosis': diagnosis
})

data.to_csv('synthetic_schizophrenia_data.csv', index=False)
print("Synthetic dataset generated and saved as 'synthetic_schizophrenia_data.csv'.")
X = data[['Age','Gender','WorkingMemoryScore','ReactionTime','PANSS_Positive','PANSS_Negative']]
Y = data['Diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled,Y)
Schizo_pred = model.predict(X_scaled)
accuracy = accuracy_score(data['Diagnosis'],Schizo_pred)
print("Accuracy :",accuracy*100)
plt.figure(figsize=(8,5))
plt.scatter(data['PANSS_Positive'],data['Diagnosis'],color = 'deeppink')
plt.plot()
plt.xlabel("Positive Symptoms")
plt.ylabel("Diagnosis")
plt.title("Positive Symptoms vs Diagnosis")
plt.grid(True)
plt.show()
