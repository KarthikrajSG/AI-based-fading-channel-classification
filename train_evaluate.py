from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data"/"channel_features.csv")
cols=[c for c in df.columns if c not in ("sample_id","channel_type","snr_db")]
X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0).values
le=LabelEncoder(); y=le.fit_transform(df.channel_type.values); snr=df.snr_db.values
tr,te=train_test_split(np.arange(len(y)),test_size=.25,random_state=42,stratify=y)
models={
 "SVM":make_pipeline(StandardScaler(),SVC(kernel="rbf",C=3)),
 "Random Forest":RandomForestClassifier(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "MLP":make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,early_stopping=True,random_state=42))
}
overall=[]; by_snr=[]
for name,m in models.items():
 m.fit(X[tr],y[tr]); p=m.predict(X[te])
 overall.append({"model":name,"accuracy":accuracy_score(y[te],p),"macro_f1":f1_score(y[te],p,average="macro")})
 for level in sorted(np.unique(snr)):
  mask=snr[te]==level
  by_snr.append({"model":name,"snr_db":int(level),"accuracy":accuracy_score(y[te][mask],p[mask]),"macro_f1":f1_score(y[te][mask],p[mask],average="macro")})
pd.DataFrame(overall).to_csv(ROOT/"outputs"/"overall_model_comparison.csv",index=False)
pd.DataFrame(by_snr).to_csv(ROOT/"outputs"/"performance_by_snr.csv",index=False)
print(pd.DataFrame(overall).to_string(index=False))
