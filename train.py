"""
=====================================
CareerScan AI — train.py
=====================================
"""

import pandas as pd #data ko tabular form mai handle karna ka liya hoti hai 
import numpy as np # numerical python for mathematical calculation and handle arrays
import matplotlib.pyplot as plt # for visulization graphs and charts 
import seaborn as sns 
import joblib #model ko save aur load karna ka liya use hoti hai 
import pickle # objects ko save karna ka liya hoti hai like pipeline 
from sklearn.model_selection import train_test_split #sklearn model banany ka liys use hoti hai aur yhan ya dataset ko split karti hai train aur test data mai
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder #StandardScaler data ko ak sacle/range mai la ka ata hai  aur labelencoder text ko number mai convert karta hai 

from sklearn.metrics import (
    accuracy_score,        # Overall sahi predictions ka ratio — basic overview ke liye
    f1_score,              # Precision + Recall ka balance — imbalanced data mein sabse useful
    roc_auc_score,         # Model kitna acha positive/negative alag kare — 1.0 = perfect
    recall_score,          # Actual positives mein se kitne pakde (miss na ho koi)
    precision_score,       # Jo positive bola unme se kitne sach mein positive (false alarm na ho)
    classification_report, # Har class ki precision/recall/f1 ek saath text mein
    confusion_matrix,      # TP, FP, TN, FN ka raw matrix — errors kahan hain
    ConfusionMatrixDisplay # Confusion matrix ko matplotlib se plot karne ke liye
)

from xgboost import XGBClassifier

# ═══════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────

df = pd.read_csv("ai_job_impact.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

# ─────────────────────────────────────────
# STEP 2: MISSING VALUES
# ─────────────────────────────────────────

print("\nMissing Values:")
print(df.isnull().sum()) 

number_cols = df.select_dtypes(include='number').columns
df[number_cols] = df[number_cols].fillna(df[number_cols].mean())

text_cols = df.select_dtypes(include='object').columns
df[text_cols] = df[text_cols].fillna(df[text_cols].mode().iloc[0]) #agr mode ki do rowa ah rhy hon tw ya first row select kary ga missing values ko fill karnay ka liaya

print(" Missing values are handled ")

# ─────────────────────────────────────────
# STEP 3: REMOVE DUPLICATES
# ─────────────────────────────────────────

print("\nDuplicates Before:", df.duplicated().sum())

df = df.drop_duplicates()
print (df.shape)
print("Duplicates After:", df.duplicated().sum())

# ─────────────────────────────────────────
# STEP 4: FIX DATA TYPES
# ─────────────────────────────────────────

df['Age'] = df['Age'].astype(int) #50.0 -> 50

df['Years_Experience'] = df['Years_Experience'].astype(int) #10.0 -> 10

print("Data types fixed")

# ─────────────────────────────────────────
# STEP 5: TEXT CLEANING
# ─────────────────────────────────────────

for col in text_cols:

    df[col] = df[col].str.strip() # extra spaces removed "he  lp" ->help

    df[col] = df[col].str.title() # aGHY ->Aghy

print("Text is cleaned")

# ─────────────────────────────────────────
# STEP 6: OUTLIERS
# ─────────────────────────────────────────

cols_to_check = [
    'Age',
    'Years_Experience',
    'Salary_Before_AI',
    'Work_Hours_Per_Week',
    'Job_Satisfaction',
    'Productivity_Change_%'
]  #colums list 

for col in cols_to_check:

    Q1 = df[col].quantile(0.25) #1st quartile

    Q3 = df[col].quantile(0.75)# 3rd

    IQR = Q3 - Q1 # tell the middle spreadout of data 

    lower = Q1 - 1.5 * IQR # issa lower values outliars mai jati hain

    upper = Q3 + 1.5 * IQR  # is sa upper values ouliers mai jati han

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    print(f"{col}: {outliers} outliers")

    df[col] = df[col].clip(lower, upper) # small values become lower limit and large values become upper limit

print("Outliers are handled")

# ─────────────────────────────────────────
# STEP 7: DROP USELESS COLUMNS
# ─────────────────────────────────────────

df = df.drop(
    columns=['Employee_ID', 'Salary_After_AI'], 
    errors='ignore'
)

print(" Unnecessary columns removed ")

# ─────────────────────────────────────────
# STEP 8: ENCODING
# ─────────────────────────────────────────

encode_cols = [
    'Gender', 
    'Education_Level',
    'Industry',
    'Job_Role',
    'AI_Adoption_Level',
    'Automation_Risk',
    'Upskilling_Required',
    'Remote_Work'
]

# Har column ka alag encoder
col_encoders = {} # save all the encoded columns

for col in encode_cols:
  
    le_col = LabelEncoder() # convert text to number

    df[col] = le_col.fit_transform(df[col]) # learn -> apply

    col_encoders[col] = le_col 

    print(f"{col} encoded")

# Target encoder
le_target = LabelEncoder()

df['Job_Status'] = le_target.fit_transform(df['Job_Status'])

print("Job_Status classes:")
print(list(le_target.classes_))

print("Encoding complete")

# ─────────────────────────────────────────
# STEP 9: FEATURES & TARGET
# ─────────────────────────────────────────

X = df.drop('Job_Status', axis=1)

y = df['Job_Status']

# ─────────────────────────────────────────
# STEP 10: TRAIN TEST SPLIT
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% test data aur 80% train data
    random_state=42 #same result repeat karna
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print("Train Test Split complete")

# ─────────────────────────────────────────
# STEP 11: SCALING
# ─────────────────────────────────────────

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling complete")

# ═══════════════════════════════════════════════════════════
# PHASE 2 — EDA PLOTS
# ═══════════════════════════════════════════════════════════

numeric_cols = [
    'Age',
    'Years_Experience',
    'Salary_Before_AI',
    'Work_Hours_Per_Week',
    'Job_Satisfaction',
    'Productivity_Change_%'
]


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, col in zip(axes.flatten(), numeric_cols): #2d->1d

    ax.hist(
        df[col],
        bins=30,
        color='steelblue',
        edgecolor='black'
    )

    ax.set_title(col) #column name =title of histogram

plt.tight_layout() #avoid overlapping of graphs by automatically adjusting the space 

plt.savefig('histograms.png', dpi=150)

plt.close()

# Heatmap
corr = df.select_dtypes(include='number').corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True, #true values show karay ga box mai
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig('heatmap.png', dpi=150)

plt.close()

categorical_plots = ['Gender', 'Industry', 'Job_Role','Education_Level']
for col in categorical_plots:
    counts = df[col].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, color='steelblue')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Distribution of {col}')
    if col in ['Industry', 'Job_Role']:
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'bar_{col}.png', dpi=300)
    plt.close()
    print(f"✅ Saved: bar_{col}.png")


# Box plot WITHOUT Salary_After_AI (data leakage)
sns.boxplot(x='Job_Status', y='Salary_Before_AI', data=df)
plt.title('Salary Before AI by Job Status Changes')
plt.tight_layout()
plt.savefig('Salary_Before_AI_by_Job_Status.png', dpi=300)
plt.close()
print("✅ Saved: Salary_Before_AI_by_Job_Status.png")

sns.boxplot(x='Job_Status', y='Age', data=df)
plt.title('Age by Job Status Changes')
plt.tight_layout()
plt.savefig('Age_by_Job_Status.png', dpi=300)
plt.close()
print("✅ Saved: Age_by_Job_Status.png")

sns.boxplot(x='Job_Status', y='Years_Experience', data=df)
plt.title('Years of Experience by Job Status Changes')
plt.tight_layout()
plt.savefig('Years_Experience_by_Job_Status.png', dpi=300)
plt.close()
print("✅ Saved: Years_Experience_by_Job_Status.png")

# AI Adoption vs Job Status
sns.countplot(x='AI_Adoption_Level', hue='Job_Status', data=df)
plt.title('Job Status Variation Across AI Adoption Levels')
plt.tight_layout()
plt.savefig('AI_Adoption_Level_VS_Job_Status.png', dpi=300)
plt.close()
print("✅ Saved: AI_Adoption_Level_VS_Job_Status.png")

# Automation Risk vs Job Status
sns.countplot(x='Automation_Risk', hue='Job_Status', data=df)
plt.title('Job Status Variation Across Automation Risk Levels')
plt.tight_layout()
plt.savefig('Automation_Risk_VS_Job_Status.png', dpi=300)
plt.close()
print("✅ Saved: Automation_Risk_VS_Job_Status.png")

print("EDA plots complete")


# ═══════════════════════════════════════════════════════════
# PHASE 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════


# ─────────────────────────────────────────
# STEP 1: Replaced INDEX
# ─────────────────────────────────────────

replaced_idx = list(le_target.classes_).index('Replaced')

print(f"Replaced Index: {replaced_idx}") # 1

# 0-> modified 1->replaced 2->unchanged
# ─────────────────────────────────────────
# STEP 2: CLASS WEIGHTS
# ─────────────────────────────────────────

class_counts = pd.Series(y_train).value_counts().sort_index() # count how many time each class come in taning data and arrange in oredr 
print(class_counts)
total = len(y_train) #target train data ki length 

custom_weights = {}

for idx, count in class_counts.items():

    cls_name = le_target.classes_[idx] #index pa konsi  value hai  0,1,2

    if cls_name == 'Replaced':

        custom_weights[int(idx)] = total / (0.6 * count)

    else:

        custom_weights[int(idx)] = total / (3 * count)

print("Class Weights:")
print(custom_weights)

# ─────────────────────────────────────────
# STEP 3: MODELS
# ─────────────────────────────────────────

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000, #maximum attemps model ko train karna ka liya 
        class_weight=custom_weights, # imbalance data ko balance karta hai 
        C=0.5, # model ko overfitting sa bacahta hai 
        random_state=42 # same result repeat karna 
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight=custom_weights,
        max_depth=5, #tree kitna deep ho sakta hai 
        min_samples_leaf=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        class_weight=custom_weights,
        n_estimators=300, # how many trees 
        max_depth=8, 
        min_samples_leaf=5,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1, # model kitna speed sa learn kary
        objective='multi:softprob', # output probabilities mai deta hai 
        num_class=len(le_target.classes_), #total number of classes
        random_state=42 # same result repeat karna 
    )
}

# ─────────────────────────────────────────
# STEP 4: VARIABLES
# ─────────────────────────────────────────

best_model = None

best_model_name = ""

best_replaced_recall = 0

results = {}

# ─────────────────────────────────────────
# STEP 5: TRAINING LOOP
# ─────────────────────────────────────────

for name, model in models.items():

    print(f"\n{'='*50}") #=============

    print(f"TRAINING: {name}")

    # Logistic Regression scaled data use karega
    if name == "Logistic Regression":

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        y_proba = model.predict_proba(X_test_scaled)

    else:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred) # ->kitni prediction shi hai

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )
   #F1 Score balance karta hai:Precision Recall

    recalls = recall_score(
        y_test,
        y_pred,
        average=None
    ) #0.30
     #Actual positive cases mein se kitne correctly predict hue.
    precisions = precision_score(
        y_test,
        y_pred,
        average=None
    )
    #Jo positive predict kiya unmein kitne actually positive the.

    replaced_recall = recalls[replaced_idx]

    replaced_precision = precisions[replaced_idx]

    try:

        roc = roc_auc_score(
            y_test,
            y_proba,
            multi_class='ovr'
        )

    except:

        roc = None

    # Save Results
    results[name] = {

        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': roc,
        'replaced_recall': replaced_recall,
        'replaced_precision': replaced_precision,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model
    }

    # Print
    print(f"Accuracy : {acc:.4f}") #-> 4 decimal places

    print(f"F1 Score : {f1:.4f}")

    if roc is not None:

        print(f"ROC AUC : {roc:.4f}")

    # Classification Report
    print("\nClassification Report:\n")

    print(classification_report(
        y_test,
        y_pred,
        target_names=le_target.classes_
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le_target.classes_, # modified ,replaced aur uncahged 
        yticklabels=le_target.classes_
    )

    plt.title(f'Confusion Matrix — {name}')

    plt.tight_layout()

    plt.savefig(
        f'confusion_matrix_{name.replace(" ","_")}.png',
        dpi=150
    )

    plt.close()

    # Feature Importance
    if hasattr(model, 'feature_importances_'):

        imp_df = pd.DataFrame({

            'feature': X.columns,
            'importance': model.feature_importances_

        })

        imp_df = imp_df.sort_values(
            'importance',
            ascending=False
        )

        plt.figure(figsize=(9,5))

        sns.barplot(
            data=imp_df.head(10),
            x='importance',
            y='feature'
        )

        plt.title(f'Top Features — {name}')

        plt.tight_layout()

        plt.savefig(
            f'feature_importance_{name.replace(" ","_")}.png',
            dpi=150
        )

        plt.close()

    # Best Model
    if replaced_recall > best_replaced_recall: # ->30 >0

        best_replaced_recall = replaced_recall

        best_model = model

        best_model_name = name

# ═══════════════════════════════════════════════════════════
# PHASE 6 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60) #========

print("FINAL SUMMARY")

print("=" * 60)

for name, res in sorted(
    results.items(),
    key=lambda x: x[1]['replaced_recall'],
    reverse=True
):

    tag = " <-- selected" if name == best_model_name else ""

    print(
        f"{name:20s} | "
        f"Recall: {res['replaced_recall']*100:.1f}% | "
        f"Precision: {res['replaced_precision']*100:.1f}% | "
        f"F1: {res['f1_score']:.4f}"
        f"{tag}"
    )

print(f"\n🔥 BEST MODEL: {best_model_name}")

# ═══════════════════════════════════════════════════════════
# PHASE 7 — SAVE PIPELINE
# ═══════════════════════════════════════════════════════════

pipeline = {
    'model': best_model,
    'scaler': scaler,
    'label_encoder': le_target,
    'col_encoders': col_encoders,
    'feature_columns': list(X.columns),
    'best_model_name': best_model_name,
    'X_train': X_train.values  # Save raw training data (not scaled)
}

with open('ai_impact_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

joblib.dump(best_model, 'best_model.pkl')

print("✅ Pipeline Saved  data")



