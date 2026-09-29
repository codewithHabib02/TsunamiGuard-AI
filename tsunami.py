import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,roc_auc_score

df=pd.read_csv('earthquake_data_tsunami.csv')

print("\n" + "=" * 70)
print("Head Of The Dataset:")
print(df.head())
print("\n" + "=" * 70)


print("\n" + "=" * 70)
print("Info Of The Dataset:")
print(df.info())
print("\n" + "=" * 70)


print("\n" + "=" * 70)
print("Missing Values Of The Dataset:")
print(df.isnull().sum())
print("\n" + "=" * 70)

print("\n" + "=" * 70)
print("Total Missing Values Of The Dataset:")
print(df.isnull().sum().sum())
print("\n" + "=" * 70)

print("\n" + "=" * 70)
print("nPecentage Of Missing Values Of The Dataset:")
print(df.isnull().sum()/len(df)*100)
print("\n" + "=" * 70)

print("\n" + "=" * 70)
print("Columns Of The Dataset:")
print(df.columns.tolist())
print("\n" + "=" * 70)

print("\n" + "=" * 70)
print("Dtypes Of The Dataset:")
print(df.dtypes)
print("\n" + "=" * 70)

print("\n" + "=" * 70)
print("Duplicates Of The Dataset:")
print(df.duplicated().sum())
print("\n" + "=" * 70)

print("\n" + "=" * 70)
print("Description Of The Dataset:")
print(df.describe())
print("\n" + "=" * 70)


categroical_cols=df.select_dtypes(include=['category','object']).columns.tolist()
for c in categroical_cols:
    print("\n" + "=" * 70)
    print(f"Categorical Analysis {c}")
    print("\n" + "=" * 70)
    print(c, df[c].value_counts(),df[c].nunique(),df[c].unique()[:10])
    print(categroical_cols)

numeric_cols=df.select_dtypes(include=['number']).columns.tolist()
for col in numeric_cols:
     print("\n" + "=" * 70)
     print(f"Numeric and Outlier Analysis {col}")
     print("\n" + "=" * 70)

     Q1=df[col].quantile(0.25)
     Q3=df[col].quantile(0.75)
     IQR=Q3-Q1

     lower=Q1 - 1.5 * IQR
     upper=Q3 + 1.5 * IQR

     outliers=(
          (df[col] < lower) |
          (df[col] > upper)
     ).sum()

     print(
          f" Count            :{df[col].count()}\n"
          f"Missing_values    :{df[col].isnull().sum()}\n"
          f"Mean              :{df[col].mean()}\n"
          f"Median            :{df[col].median()}\n"
          f"Max               :{df[col].max()}\n"
          f"Min               :{df[col].min()}\n"
          f"Std               :{df[col].std()}\n"
          f"Outlier           :{outliers}"

         )
df.columns=df.columns.str.strip()     

x=df.drop('tsunami',axis=1)
y=df['tsunami']

numeric=x.select_dtypes(include=['number']).columns.tolist()
preprocessor=ColumnTransformer(transformers=[
     ("num",
        Pipeline([
           ("imput",SimpleImputer(strategy='mean')),
           ("scaler", StandardScaler()),
      ]),numeric

      )
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



models={
     "SVC": SVC(),
     "Tree": DecisionTreeClassifier(),
     "Reg":  LogisticRegression(),
     "Ridge": RidgeClassifier(),
     "RandomR": RandomForestClassifier(random_state=42)
     
}

restult=[]
for name, model in models.items():

     pip=Pipeline(steps=[
          ("prepro",preprocessor),
          ("model", model)
     ])

     pip.fit(x_train,y_train)
     prediction=pip.predict(x_test)

     restult.append({

          "Model":     name,
          "Accuracy":  accuracy_score(y_test,prediction),
          "Precision": precision_score(y_test,prediction),
          "Recall":    recall_score(y_test,prediction),
          "F1":        f1_score(y_test,prediction),
        
        
          
     })

restult_df=pd.DataFrame(restult)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(restult_df)





fmodel=[]
pip=Pipeline(steps=[
          ("prepro",preprocessor),
          ("model", RandomForestClassifier())
     ])


param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}

grid=GridSearchCV(
     pip,
     param_grid,
     cv=5,
     scoring='f1'
)

grid.fit(x_train,y_train)

print(
     f"best_param:  {grid.best_params_}\n"
     f"best_score:  {grid.best_score_}\n"
     f"best_estimator: {grid.best_estimator_}"
)

fmodel=[]
final_model=Pipeline(steps=[
          ("prepro",preprocessor),
          ("model", RandomForestClassifier(
                max_depth=20,
                max_features="sqrt",
                min_samples_leaf=2,
                min_samples_split=10,
                n_estimators=200,
                random_state=42
          ))
     ])

final_model.fit(x_train,y_train)

x_trained_pred=final_model.predict(x_train)
x_tested_pred=final_model.predict(x_test)
train_acc=accuracy_score(y_train,x_trained_pred)
test_acc=accuracy_score(y_test,x_tested_pred)
print(
     f"Trained: {train_acc: .2f}\n"
     f"Tested : {test_acc: .2f}"
)

labels=["Training", "Testing"]
scores=[train_acc,test_acc]
plt.bar(labels,scores)
plt.xlim(0,1)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.show()

prediction=final_model.predict(x_test)
fmodel.append({

          "Model":     "RandomforestClassifier",
          "Accuracy":  accuracy_score(y_test,prediction),
          "Precision": precision_score(y_test,prediction),
          "Recall":    recall_score(y_test,prediction),
          "F1":        f1_score(y_test,prediction),
          "Roc":       roc_auc_score(y_test,final_model.predict_proba(x_test)[:,1])
        
          
     })

fmodel_df=pd.DataFrame(fmodel)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(fmodel_df)

feature=final_model.named_steps["prepro"].get_feature_names_out()
cofficion=final_model.named_steps['model'].feature_importances_

imp=pd.DataFrame({
     "feature": feature,
     "importance": cofficion * 100
}).sort_values(by="importance", ascending=False)

print(imp.head(10))

perm=permutation_importance(
     final_model,
     x_test,
     y_test,
     n_repeats=5,
     random_state=42,
     scoring='accuracy'
)

mutation=pd.DataFrame({
     "feature": x_test.columns,
     "importance": perm.importances_mean * 100
}).sort_values(by='importance',ascending=False)

print(mutation.head(10))


selected_cols = [
    "Year",
    "dmin",
    "longitude",
    "nst",
    "latitude",
    "depth",
    "gap",
    "sig"
] 

x=df[selected_cols]
y=df['tsunami']

numerical=x.select_dtypes(include=['number']).columns.tolist()
preprocessor=ColumnTransformer(transformers=[
     ("num",
        Pipeline([
           ("imput",SimpleImputer(strategy='mean')),
           ("scaler", StandardScaler()),
      ]),numerical

      )
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

fmodel=[]
best_model=Pipeline(steps=[
          ("prepro",preprocessor),
          ("model", RandomForestClassifier(
                max_depth=20,
                max_features="sqrt",
                min_samples_leaf=2,
                min_samples_split=10,
                n_estimators=200,
                random_state=42
          ))
     ])

best_model.fit(x_train,y_train)
import joblib
joblib.dump(best_model,"tsunami_model.pkl")

x_trained_pred=best_model.predict(x_train)
x_tested_pred=best_model.predict(x_test)
train_acc=accuracy_score(y_train,x_trained_pred)
test_acc=accuracy_score(y_test,x_tested_pred)
print(
     f"Trained: {train_acc: .2f}\n"
     f"Tested : {test_acc: .2f}"
)

labels=["Training", "Testing"]
scores=[train_acc,test_acc]
plt.bar(labels,scores)
plt.xlim(0,1)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.close()

prediction=best_model.predict(x_test)
fmodel.append({

          "Model":     "RandomforestClassifier",
          "Accuracy":  accuracy_score(y_test,prediction),
          "Precision": precision_score(y_test,prediction),
          "Recall":    recall_score(y_test,prediction),
          "F1":        f1_score(y_test,prediction),
          "Roc":       roc_auc_score(y_test,best_model.predict_proba(x_test)[:,1])
        
          
     })

fmodel_df=pd.DataFrame(fmodel)
print(fmodel_df)

