import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("Retail_Sales_Dataset.csv")

print(df.head())
print(df.info())
print(df.tail())

X=df[["Age","Quantity","Price per Unit"]]

y=(df['Total Amount'] >=500) .astype(int)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))




















