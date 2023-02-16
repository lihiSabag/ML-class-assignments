import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading the dataset
play_tennis = pd.read_csv("PlayTennis.csv")
print(play_tennis.head())

#Converting the properties to numerical values
number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])

features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"
print(play_tennis.head())
features_train, features_test, target_train, target_test = train_test_split(play_tennis[features], play_tennis[target],

                                                                            test_size = 0.33, random_state = 54)
#training
model = GaussianNB()
model.fit(features_train, target_train)

#testing
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)

print(f"accuracy: {accuracy}")
print(f"model predict for [Rain,Mild,High,Weak] is : {model.predict([[1,2,0,1]])}")