import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
model = joblib.load("model.pkl")
test_df = pd.read_csv("Iris.csv")
test_df=test_df.drop("Id",axis=1)
x = test_df.drop("Species", axis = 1)
y = test_df["Species"]
pred = model.predict(x)
acc = accuracy_score(pred, y)
print(acc)
assert len(pred) == len(y)
#If first statement will fail - the program will give error and it will not move to next assert statement
print("Test 1 passed")
#len - length
# assert acc>=0.50
assert acc>=0.99
print("All tests passed")