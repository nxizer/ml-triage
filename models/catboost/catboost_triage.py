import pandas as pd
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

train_path = "/home/nxizer/workdir/ml-triage/dataset/train.csv"
dev_path = "/home/nxizer/workdir/ml-triage/dataset/dev.csv"

train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)

print("Train label distribution (before):")
print(train_df['label'].value_counts())

df_class0 = train_df[train_df['label'] == 0]
df_class1 = train_df[train_df['label'] == 1]

df_class0 = train_df[train_df['label'] == 0]
df_class1 = train_df[train_df['label'] == 1]

n_samples = min(len(df_class0), len(df_class1))

# выравнивается label=0 по количеству label=1
df_class0_downsampled = resample(df_class0,
                                 replace=False,
                                 n_samples=n_samples,
                                 random_state=42)

df_class1_final = df_class1

print("\nDev label distribution (before):")
print(dev_df['label'].value_counts())

# dev_df = dev_df[dev_df['label'] == 1].reset_index(drop=True)
# print("Dev shape (after, only label==1):", dev_df.shape)

dev_class0 = dev_df[dev_df['label'] == 0]
dev_class1 = dev_df[dev_df['label'] == 1]

n_dev_samples = min(len(dev_class0), len(dev_class1))

dev_class0_downsampled = resample(dev_class0,
                                  replace=False,
                                  n_samples=n_dev_samples,
                                  random_state=42)

dev_class1_final = dev_class1

dev_df = pd.concat([dev_class0_downsampled, dev_class1_final]).sample(frac=1, random_state=42).reset_index(drop=True)
train_df = pd.concat([df_class0_downsampled, df_class1_final]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nTrain label distribution (after):")
print(train_df['label'].value_counts())
print("\nDev label distribution (after):")
print(dev_df['label'].value_counts())

train_df['text_combined'] = (
    train_df['bug_type'].fillna('') + ' ' +
    train_df['trace'].fillna('') + ' ' +
    train_df['bug_function'].fillna('') + ' ' +
    train_df['functions'].fillna('')
)
dev_df['text_combined'] = (
    dev_df['bug_type'].fillna('') + ' ' +
    dev_df['trace'].fillna('') + ' ' +
    dev_df['bug_function'].fillna('') + ' ' +
    dev_df['functions'].fillna('')
)

y_train = train_df['label']
y_dev = dev_df['label']

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=100
)

X_train_text = vectorizer.fit_transform(train_df['text_combined'])
X_dev_text   = vectorizer.transform(dev_df['text_combined'])

train_pool = Pool(data=X_train_text, label=y_train)
dev_pool   = Pool(data=X_dev_text,   label=y_dev)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=None,
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42,
    thread_count=-1
)

model.fit(train_pool, eval_set=dev_pool)

dev_preds = model.predict(dev_pool)
accuracy = accuracy_score(y_dev, dev_preds)
print("Accuracy:", accuracy)
