# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:35:55 2016

@author: Team 6
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 22:45:13 2016

@author: Team 6
10-K Classifier
"""
#Importing required modules
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk import stem
from sklearn.feature_selection import SelectPercentile, f_classif
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt


#Creates the master dataframe from the text files and assigns them labels
def get_dataset(path):
    dataset=[]
    try:
        os.chdir(path)
    except:
        print "Incorrect path name!"
    
    for filename in os.listdir(path):
        f=open(filename,'r')
        split=filename.split("_")
        Year=split[4]
        Name=split[7]
        if re.search("POS",filename):
            if(re.search("pos",filename)):
                dataset.append([Name,Year,f.read(),"pos","pos"])
            else:
                dataset.append([Name,Year,f.read(),"pos","neg"])
        else:
             if(re.search("pos",filename)):
                dataset.append([Name,Year,f.read(),"neg","pos"])
             else:
                dataset.append([Name,Year,f.read(),"neg","neg"])
    dataset=pd.DataFrame(dataset)
    dataset.columns = ['Name','Year','MDA_Text','Sentiment','Old_Sentiment']
    return dataset

#Splitting into training and testing set
def split(df,test_ratio):
    return train_test_split(df, test_size = test_ratio, stratify = df['Sentiment'],random_state=100)
    
#Function to stem words a string    
def stemming(x):
    stemmer = stem.SnowballStemmer("english")
    words=x.split()
    doc=[]
    for word in words:
        word=stemmer.stem(word)
        doc.append(word)
    return " ".join(doc)

#Function to remove all non-characters from MD&As
def preprop(dataset):
    dataset['MDA_Text']=dataset['MDA_Text'].str.replace("[^a-zA-Z]", ' ')
    return dataset

#Function to create features of total positive and total negative words based on Loughran McDonald Dictionary
def count_fin_words(lmd,dataset):
    #Modifying the Dictionary  
    lmd=lmd[['Word','Positive','Negative']]
    lmd['Sum']=lmd['Positive']+lmd['Negative']
    lmd=lmd[lmd.Sum != 0]
    lmd=lmd.drop(['Sum'],axis=1)
    lmd.loc[lmd['Positive']>0, 'Positive'] = 1
    lmd.loc[lmd['Negative']>0, 'Negative'] = -1
    lmd['Word']=lmd['Word'].str.lower()
    #Counting the words in the MDA
    tf = CountVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(dataset['MDA_Text'].values)
    feature_names = tf.get_feature_names() 
    tfidf_array = tfidf_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_array)
    tfidf_df.columns = [i.lower() for i in feature_names] 
    tfidf_df = tfidf_df.T 
    tfidf_df['Word']=tfidf_df.index
    #Merging the results
    result_df = pd.merge(tfidf_df, lmd, how='inner',left_on='Word',right_on='Word')
    col_list=list(result_df)
    result_df_pos=result_df[result_df.Positive==1]
    result_df_neg=result_df[result_df.Negative==-1]
    result_df[col_list[0:len(dataset)]].sum(axis=0)
    #Counting the positive and negative words in a financial context per document
    pos_words_sum=result_df_pos[col_list[0:len(dataset)]].sum(axis=0)
    neg_words_sum=result_df_neg[col_list[0:len(dataset)]].sum(axis=0)
    #Adding new features to the master dataframe
    dataset['Tot_pos']=pos_words_sum.values
    dataset['Tot_neg']=neg_words_sum.values
    return dataset

#Function to create polarity score feature
def polarity_score(dataset):
    polarity=[]
    polarity_score=[]
    for mda,sentiment in zip(dataset['MDA_Text'].values,dataset['Sentiment'].values):
        blob=TextBlob(mda)
        score = blob.sentiment.polarity
        polarity.append([score,sentiment])
        polarity_score.append(score)
    dataset['Polarity']=polarity_score
    return dataset

#Function to add features to the train and test set based on vectorizer
def vect_features(vectorizer,train,test):
    features_train_transformed = vectorizer.fit_transform(train['MDA_Text'].values)
    feature_names = vectorizer.get_feature_names()
    features_train_transformed = features_train_transformed.toarray()
    train_data = pd.DataFrame(features_train_transformed)
    train_data.columns = feature_names
    train=pd.concat([train,train_data],axis=1)
    features_test_transformed = vectorizer.transform(test['MDA_Text'].values)
    features_test_transformed = features_test_transformed.toarray()
    test_data = pd.DataFrame(features_test_transformed)
    test_data.columns = feature_names
    test=pd.concat([test,test_data],axis=1)
    return train,test

#Function to create Classification Report   
def report(test,predictions):
   print pd.crosstab(test['Sentiment'], predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
   a=accuracy_score(test['Sentiment'],predictions)
   p=precision_score(test['Sentiment'],predictions, pos_label = "pos")
   r=recall_score(test['Sentiment'].values,predictions, pos_label = "pos")
   f=f1_score(test['Sentiment'].values,predictions, pos_label = "pos")
   print "Accuracy = ",a,"\nPrecision =",p,"\nRecall = ",r,"\nF-Score = ",f 

#Function to create models and print accuracies
def model(classifier,train,test,column):
    targets = train['Sentiment'].values
    train_data=train.values
    predictors = train_data[0:,column:]
    classifier.fit(predictors,targets)
    test_data=test.values
    predictions=classifier.predict(test_data[0:,column:])
    report(test,predictions)
    return predictions


#Reading the Loughran McDonald Dictionary    
os.chdir("D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 7")
lmd = pd.read_excel("LoughranMcDonald_MasterDictionary_2014.xlsx")      

#Defining the path    
path = "D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 8\MDAs"

#Creating the master dataframe
dataset=get_dataset(path)

#Counting class frequency
dataset['Sentiment'].value_counts()
dataset['Sentiment']=dataset['Sentiment'].astype('category')

#Checking empty MDAs
count=0
for mda in dataset['MDA_Text'].values:
    if len(mda)<500:
        print count
    count=count+1

#Preprocessing the master dataframe
dataset=preprop(dataset)

#Adding total positive and total negative words based on Loughran McDonald Dictionary to the master dataframe
dataset=count_fin_words(lmd,dataset)

#Creating polarity score feature
dataset=polarity_score(dataset)

#Stemming the MD&A Text
stemmer = stem.SnowballStemmer("english")
dataset['MDA_Text']=dataset['MDA_Text'].apply(stemming)

#Proportion of Postive and Negative Features
dataset['Prop_pos']=dataset['Tot_pos']/(dataset['Tot_pos']+dataset['Tot_neg'])
dataset['Prop_neg']=dataset['Tot_neg']/(dataset['Tot_pos']+dataset['Tot_neg'])
dataset['Diff']=dataset['Tot_pos']-dataset['Tot_neg']

#Rearranging Columns
cols = list(dataset.columns.values)
dataset = dataset[['MDA_Text','Sentiment','Tot_pos','Tot_neg','Old_Sentiment','Diff','Polarity','Prop_pos','Prop_neg']]

#Pickling dataset
os.chdir("D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 9")
lmd.to_pickle("lmd.pkl")
dataset.to_pickle("dataset_nostem_final.pkl")

#Reading Pickle
os.chdir("D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 9")
dataset = pd.read_pickle("dataset_nostem_final.pkl")


#Vocabulary
fin_vocab=lmd['Word'].tolist()

#Splitting to training and testing
train, test = split(dataset,100)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)


#Baseline Accuracy
report(dataset,dataset['Old_Sentiment'].values)
report(test,test['Old_Sentiment'].values)

#Model 1 - Baseline Model
#Algorithm: Bernoulli Naive Bayes 
#Features: Contains all words
vectorizer_1 = CountVectorizer(stop_words='english',max_features=5000)
train_1,test_1 = vect_features(vectorizer_1,train,test)
classifier = BernoulliNB(fit_prior=False)
predictions = model(classifier,train_1,test_1,8)
train_1.columns.values

#Model 2 
#Algorithm: Bernoulli Naive Bayes 
#Features: Contains top 50 words
vectorizer_2 = CountVectorizer(stop_words='english',max_features=50)
train_2,test_2 = vect_features(vectorizer_2,train,test)
classifier = BernoulliNB(fit_prior=False)
predictions = model(classifier,train_2,test_2,9)
train_2.columns.values

#Model 3 
#Algorithm: Multinomial Naive Bayes 
#Features: CountVectorizer of all words
vectorizer_3 = CountVectorizer(stop_words='english',max_features = 5000)
train_3,test_3 = vect_features(vectorizer_3,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_3,test_3,8)
train_3.columns.values

#Model 4 
#Algorithm: Multinomial Naive Bayes 
#Features: CountVectorizer of top 50 words
vectorizer_4 = CountVectorizer(stop_words='english',max_features = 50)
train_4,test_4 = vect_features(vectorizer_4,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_4,test_4,8)
train_4.columns.values

#Model 5
#Algorithm: Multinomial Naive Bayes 
#Features: CountVectorizer of top 100 words
vectorizer_5 = CountVectorizer(stop_words='english',max_features = 100)
train_5,test_5 = vect_features(vectorizer_5,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_5,test_5,8)
train_5.columns.values

#Model 6
#Algorithm: Multinomial Naive Bayes 
#Features: CountVectorizer of top 25 words
vectorizer_6 = CountVectorizer(stop_words='english',max_features = 25)
train_6,test_6 = vect_features(vectorizer_6,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_6,test_6,8)
train_6.columns.values

#Model 7
#Algorithm: Multinomial Naive Bayes
#Features: CountVectorizer of only top 50 words and 2-grams
vectorizer_7 = CountVectorizer(stop_words='english',max_features=50,ngram_range=(1,2))
train_7,test_7 = vect_features(vectorizer_7,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_7,test_7,8)
train_7.columns.values

#Model 8
#Algorithm: Multinomial Naive Bayes
#Features: CountVectorizer of only top 500 words and 2-grams
vectorizer_8 = CountVectorizer(stop_words='english',max_features=500,ngram_range=(1,2))
train_8,test_8 = vect_features(vectorizer_8,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_8,test_8,8)
train_8.columns.values

#Model 9
#Algorithm: Multinomial Naive Bayes
#Features: CountVectorizer of only top 100 words and 2-grams
vectorizer_9 = CountVectorizer(stop_words='english',max_features=100,ngram_range=(1,2))
train_9,test_9 = vect_features(vectorizer_9,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_9,test_9,8)
train_9.columns.values

#Model 10
#Algorithm: Multinomial Naive Bayes
#Features: TfidfVectorizer of only top 500 words and 2-grams
vectorizer_10 = TfidfVectorizer(sublinear_tf=True, stop_words='english',max_features=500,ngram_range=(1,2))
train_10,test_10 = vect_features(vectorizer_10,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_10,test_10,8)
train_10.columns.values

#Model 11
#Algorithm: Multinomial Naive Bayes
#Features: TfidfVectorizer of only top 50 words and 2-grams
vectorizer_11 = TfidfVectorizer(sublinear_tf=True, stop_words='english',max_features=50,ngram_range=(1,2))
train_11,test_11 = vect_features(vectorizer_11,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_11,test_11,9)
train_11.columns.values

#Model 12
#Algorithm: Multinomial Naive Bayes
#Features: TfidfVectorizer of only top 500 words, 2-grams, min_df=10, max_df=0.9 
vectorizer_12 = TfidfVectorizer(sublinear_tf=True, stop_words='english',max_features=500,ngram_range=(1,2),min_df=10,max_df=0.9)
train_12,test_12 = vect_features(vectorizer_12,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_12,test_12,8)
train_12.columns.values

#Model 13
#Algorithm: Multinomial Naive Bayes
#Features: TfidfVectorizer of only top 500 words, 2-grams, min_df=10, max_df=0.9 
vectorizer_13 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=10,max_df=0.9)
train_13,test_13 = vect_features(vectorizer_13,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_13,test_13,8)
train_13.columns.values

#Model 14
#Algorithm: Random Forest
#Features: TfidfVectorizer of only top 500 words, 2-grams, min_df=10, max_df=0.9, polarity score, proportion positive, proportion negative
vectorizer_14 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=10,max_df=0.9)
train_14,test_14 = vect_features(vectorizer_14,train,test)
classifier = RandomForestClassifier(n_estimators=1000,random_state = 500)
predictions = model(classifier,train_14,test_14,6)
train_14.columns.values

#Model 15
#Algorithm: Random Forest
#Features: TfidfVectorizer of only top 500 words, 2-grams, min_df=10, max_df=0.9, polarity score, proportion positive, proportion negative, diff
vectorizer_15 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=10,max_df=0.9)
train_15,test_15 = vect_features(vectorizer_15,train,test)
classifier = RandomForestClassifier(n_estimators=1000,random_state = 500)
predictions = model(classifier,train_15,test_15,5)
train_15.columns.values

#Model 16
#Algorithm: Random Forest
#Features: TfidfVectorizer of only top 50 words, 2-grams, min_df=10, max_df=0.9, polarity score, proportion positive, proportion negative, diff, previous label
vectorizer_16 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=10,max_df=0.9)
train_16,test_16 = vect_features(vectorizer_16,train,test)
train_16=pd.concat([train_16,pd.get_dummies(train_16['Old_Sentiment'],prefix='Old_Sentiment')],axis=1)
test_16=pd.concat([test_16,pd.get_dummies(test_16['Old_Sentiment'],prefix='Old_Sentiment')],axis=1)
train_16['Old_Sentiment_pos']=train_16['Old_Sentiment_pos'].astype('category')
train_16['Old_Sentiment_neg']=train_16['Old_Sentiment_neg'].astype('category')
test_16['Old_Sentiment_pos']=test_16['Old_Sentiment_pos'].astype('category')
test_16['Old_Sentiment_pos']=test_16['Old_Sentiment_pos'].astype('category')
train_16['Sentiment']=train_16['Sentiment'].astype('category')
test_16['Sentiment']=test_16['Sentiment'].astype('category')
classifier = RandomForestClassifier(n_estimators=1000,random_state = 500)
predictions = model(classifier,train_16,test_16,5)
train_16.columns.values

#Model 17
#Algorithm: Random Forest
#Features: CountVectorizer of only top 50 financial dictionary words, 2-grams, min_df=10, max_df=0.9, polarity score, proportion positive, proportion negative, diff, previous label
train, test = split(dataset_nostem,100)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
vectorizer_17 = CountVectorizer(stop_words='english',max_features=50,min_df=10,max_df=0.9,vocabulary = fin_vocab)
train_17,test_17 = vect_features(vectorizer_17,train,test)
classifier = RandomForestClassifier(n_estimators=1000,random_state = 500)
predictions = model(classifier,train_17,test_17,5)
train_15.columns.values

#Model 18
#Algorithm: Random Forest
#Features: TfidfVectorizer of only top 50 financial dictionary words, 2-grams, min_df=10, max_df=0.9, polarity score, proportion positive, proportion negative, diff, previous label
train, test = split(dataset_nostem,100)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
vectorizer_18 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,min_df=10,max_df=0.9,vocabulary = fin_vocab)
train_18,test_18 = vect_features(vectorizer_18,train,test)
classifier = RandomForestClassifier(n_estimators=1000,random_state = 500)
predictions = model(classifier,train_18,test_18,5)


#Visualization

#Importing Modules
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go


#Reading Pickle
os.chdir("D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 9")
dataset = pd.read_pickle("dataset_nostem_final.pkl")

#Counting Documents in Every Year
dataset['Name']=dataset['Name'].str.upper()
dataset["Year"].value_counts()

#Labeling by Sub-Sector
unique_names=dataset['Name'].unique()
s=["Agricultural Products","Brewers","Distillers & Vintners","Drug Retail","Food Distributors","Food Retail","Household Products","Hypermarkets & Super Centers","Packaged Foods & Meats","Personal Products","Soft Drinks","Tobacco"]
sub=[s[8],s[6],s[8],s[7],s[9],s[8],s[5],s[8],s[7],s[8],s[3],s[10],s[8],s[2],s[5],s[6],s[8],s[6],s[8],s[3],s[6],s[8],s[5],s[10],s[8],s[8],s[11],s[10],s[2],s[11],s[2],s[8],s[3],s[11],s[8],s[8],s[0],s[2],s[10],s[10],s[2],s[7],s[1],s[10],s[10],s[1],s[11],s[11],s[3],s[7],s[10],s[10]]

#Adding Sub-Sector
i=-1
dataset['Sub-Sector']="None"
for name in unique_names:
    i=i+1
    dataset['Sub-Sector'][dataset['Name']==name]=sub[i]         
           

#Defining PCA
pca = PCA(n_components=2)

#Subsetting Dataset and removing columns not required
dataset_2014=dataset.query('Year=="2014"')
del dataset_2014['MDA_Text']
del dataset_2014['Sentiment']
del dataset_2014['Old_Sentiment']
del dataset_2014['Tot_pos']
del dataset_2014['Tot_neg']
del dataset_2014['Sub-Sector']
dataset_values_2014=dataset_2014.values

#Fitting PCA
pca_results = pca.fit_transform(dataset_values_2014[0:,2:])

#Plotting Results
names = dataset_2014['Name'].values
i=-1
traces=[]
for x1,y1 in pca_results:
    i=i+1
    trace=go.Scatter(x=x1,y=y1,mode='markers',name=names[i])
    traces.append(trace)

def biplot(score,coeff,pcax,pcay,labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5) 
        if labels is None:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()
    

names=dataset_2014.columns.values[2:]    
biplot(pca_results,pca.components_,1,2,labels=names)

