from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

main = tkinter.Tk()
main.title("Movie Recommendation System Using Sentiment Analysis From Microblogging Data")
main.geometry("1000x650")

global filename
global movies_df, ratings_df, tweets_df, users_df
textArray = []
movieArray = []
movieNames = []
global tfidf_vectorizer
global X
global sid
global pos
global neg
global neu

def getSentiment(movie,tweets_df):
    global sid
    result = "Unable to detect sentiment"
    tweet_data = ''
    for i in range(len(tweets_df)):
        mid = tweets_df[i,0]
        tweet = tweets_df[i,1]
        if mid == movie:
            tweet = re.sub('[^A-Za-z]+', ' ', tweet)
            tweet_data+=tweet+" "
    sentiment_dict = sid.polarity_scores(tweet_data.strip())
    negative = sentiment_dict['neg']
    positive = sentiment_dict['pos']
    neutral = sentiment_dict['neu']
    compound = sentiment_dict['compound']
    if compound >= 0.05 : 
        result = 'Positive' 
    elif compound <= - 0.05 : 
        result = 'Negative' 
    else : 
        result = 'Neutral'
    return result     

    
def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded')

def readDataset():
    global movies_df, ratings_df, tweets_df, users_df
    tweets_df = pd.read_csv("Dataset/tweets.csv", encoding='utf-8')
    movies_df =  pd.read_csv("Dataset/movies.csv")
    ratings_df = pd.read_csv("Dataset/ratings.csv")
    text.delete('1.0', END)
    text.insert(END,"Tweets Data\n")
    text.insert(END,str(tweets_df.head())+"\n\n")
    text.insert(END,"Movies Data\n")
    text.insert(END,str(movies_df.head())+"\n\n")
    text.insert(END,"Ratings Data\n")
    text.insert(END,str(ratings_df.head())+"\n\n")
    tweets_df = tweets_df.values

def collaborativeFilter():
    global X
    global tfidf_vectorizer
    textArray.clear()
    movieArray.clear()
    movieNames.clear()
    global movies_df
    movies_frame = movies_df.values
    for i in range(len(movies_df)):
        movie_id = movies_frame[i,0]
        movie_name = movies_frame[i,1]
        movie_type = movies_frame[i,2]
        movie_type = movie_type.replace("|"," ")
        data = movie_name+" "+movie_type
        data = data.lower()
        data = re.sub('[^A-Za-z]+', ' ', data)
        data = data.strip("\n").strip()
        textArray.append(data)
        movieArray.append(movie_id)
        movieNames.append(movie_name)

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    tfidf = tfidf_vectorizer.fit_transform(textArray).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.delete('1.0', END)
    text.insert(END,"Movies Content Based Model\n\n")
    text.insert(END,str(df.head()))
    df = df.values
    X = df[:, 0:df.shape[1]]

def contentFilter():
    global movies_df, ratings_df
    text.delete('1.0', END)
    ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')
    text.insert(END,"Movies Collaborative Model\n\n")
    text.insert(END,str(ratings_df.head()))

                
def sentimentModel():
    text.delete('1.0', END)
    global sid
    sid = SentimentIntensityAnalyzer()
    text.insert(END,"Sentiment Model Generated\n\n")
    
def getCollaborative(name,ratings_df):
    ratings_value = 0
    temp = ratings_df.values
    for i in range(len(temp)):
        if temp[i,4] == name:
            ratings_value = temp[i,2]
            break
    return ratings_value    

def recommendation():
    text.delete('1.0', END)
    global pos
    global neg
    global neu
    pos = 0
    neg = 0
    neu = 0
    global X
    global tfidf_vectorizer
    query = simpledialog.askstring("Query Dialog", "Type here desired movie details to get recommendation list", parent=main)
    if len(query) > 0:
        testData = query.lower()
        testData = testData.strip()
        testData = re.sub('[^A-Za-z]+', ' ', testData)
        testArray = []
        testArray.append(testData)
        testData = tfidf_vectorizer.transform(testArray).toarray()
        testData = testData[0]
        for i in range(len(X)):
            content_recom = dot(X[i], testData)/(norm(X[i])*norm(testData))
            if content_recom > 0:
                sentiment = getSentiment(movieArray[i],tweets_df)
                if sentiment == 'Positive':
                    pos = pos + 1
                if sentiment == "Negative":
                    neg = neg + 1
                if sentiment == 'Neutral':
                    neu = neu + 1
                text.insert(END,movieNames[i]+" Content Based Score "+str(content_recom)+"\n")
                text.insert(END,movieNames[i]+" Movie Sentiment "+sentiment+"\n")
                collaborative_filter = getCollaborative(movieNames[i],ratings_df)
                text.insert(END,movieNames[i]+" Collaborative Filter Rating "+str(collaborative_filter)+"\n\n")

def graph():
    global pos
    global neg
    global neu
    height = [pos,neg,neu]
    bars = ('Positive Sentiment','Negative Sentiment','Neutral Sentiment')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def close():
  main.destroy()
   
font = ('times', 16, 'bold')
title = Label(main, text='Movie Recommendation System Using Sentiment Analysis From Microblogging Data', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Movie Tweetings Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

readButton = Button(main, text="Read & Preprocess Dataset", command=readDataset)
readButton.place(x=300,y=100)
readButton.config(font=font1)

cfButton = Button(main, text="Build Collaborative Filtering Model", command=collaborativeFilter)
cfButton.place(x=600,y=100)
cfButton.config(font=font1)

cbButton = Button(main, text="Build Content Based Model", command=contentFilter)
cbButton.place(x=10,y=150)
cbButton.config(font=font1)

smButton = Button(main, text="Build Sentiment Model", command=sentimentModel)
smButton.place(x=300,y=150)
smButton.config(font=font1)

recommendationButton = Button(main, text="Movie Recommendation using All Models", command=recommendation)
recommendationButton.place(x=600,y=150)
recommendationButton.config(font=font1)

graphButton = Button(main, text="Top 10 Movies Sentiment Graph", command=graph)
graphButton.place(x=10,y=200)
graphButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=300,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
