from flask import Flask, request, render_template
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.datasets import load_files

import pickle
from nltk.corpus import stopwords


# In[2]:


dataset = load_files(r"C:\Users\ozgur\Desktop\inputset2", encoding="utf-8")
X = dataset.data
y = dataset.target
print(y)


# In[3]:


documents = []
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
    


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=0, max_df=0.7, stop_words=stopwords.words('turkish'))
X = vectorizer.fit_transform(documents).toarray()


# In[5]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 


# In[8]:


y_pred = classifier.predict(X_test)


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[10]:


pickle.dump(classifier, open('model.pkl','wb'))


# In[11]:


model = pickle.load(open('model.pkl','rb'))


# In[12]:


y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2)) 


# In[13]:


test = "Ölçüt 2.2(c)’ye göre Program Eğitim Amaçları (PEA) programın çeşitli iç ve dış paydaşları sürece dahil edilerek belirlenmelidir. Programın eğitim amaçlarının ilk kez belirlendiği ve özdeğerlendirme raporunda belirtilen 03.06.2016 tarihli Endüstri Danışma Kurulu (EDK) ve paydaş toplantısına işverenler ve bölüm akademik kadrosu katılmış olup, mezunlar sürece dahil edilmemiştir. Ziyaret sırasında, mezunlar gemide çalıştıkları için kendilerine erişmede güçlük çekildiği, bu nedenle anket yolu ile bilgi toplanmaya çalışıldığı belirtilse de anket formlarında PEA’yı belirlemeye yönelik sorular yer almadığından bu ölçüt için _ değerlendirmesi yapılmıştır."

test = vectorizer.transform([test]).toarray()
test = tfidfconverter.transform(test).toarray()
label = classifier.predict(test)[0]
print(label)
if label == 0:
    print("Kaygı")
if label == 1:
    print("Zayıflık")
    
print(model.predict(test)[0])


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13, metric = 'minkowski')
knn.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
y_pred_knn = knn.predict(X_test)
y_pred_ham = [0 for i in range(len(y_test))]

cm = confusion_matrix(y_test, y_pred_knn)
print("KNeighborsClassifier")

print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# In[15]:


from sklearn import metrics
print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred_knn)))
print("Precision: " + str(metrics.precision_score(y_test,y_pred_knn)))
print("Recall: " + str(metrics.recall_score(y_test,y_pred_knn)))
print("F1 score: " + str(metrics.f1_score(y_test,y_pred_knn)))
print("AUC score: " + str(metrics.roc_auc_score(y_test,y_pred_knn)))
print("\n")
print(metrics.classification_report(y_test,y_pred_knn))


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=42, random_state=42) 
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rfc)
print("RandomForest")

print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rfc).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# In[17]:


print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred_rfc)))
print("Precision: " + str(metrics.precision_score(y_test,y_pred_rfc)))
print("Recall: " + str(metrics.recall_score(y_test,y_pred_rfc)))
print("F1 score: " + str(metrics.f1_score(y_test,y_pred_rfc)))
print("AUC score: " + str(metrics.roc_auc_score(y_test,y_pred_rfc)))
print("\n")
print(metrics.classification_report(y_test,y_pred_rfc))


# In[18]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_gnb)
print("GaussianNB")

print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_gnb).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# In[19]:


print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred_gnb)))
print("Precision: " + str(metrics.precision_score(y_test,y_pred_gnb)))
print("Recall: " + str(metrics.recall_score(y_test,y_pred_gnb)))
print("F1 score: " + str(metrics.f1_score(y_test,y_pred_gnb)))
print("AUC score: " + str(metrics.roc_auc_score(y_test,y_pred_gnb)))
print("\n")
print(metrics.classification_report(y_test,y_pred_gnb))


# In[20]:








app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        text = request.form['olcut']
        karar = request.form['karar']
        text = vectorizer.transform([text]).toarray()
        text = tfidfconverter.transform(text).toarray()
        
        prediction = model.predict(text)[0]

        if prediction == 0:
            strprediction = "kaygı"
        else:
            strprediction = "zayıflık"

        return render_template('predict.html', prediction=strprediction, karar=karar)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug = True)