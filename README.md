# Fake-news-detection

Goal : Build a model that can accurately detect whether a piece of news is real or fake. 

All news are not real, right? So how will you detect the fake news? The answer is Python. You will easily make a difference between real and fake news.

What is Fake News?
False information disseminated with the aim of manipulating the public

The Dataset  
The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE. 

Detecting fake news with Python
1) Clean data to get high quality data and get ready for analysis
2) Split data into training and testing sets
3) Build a TfidfVectorizer 
4) Initialize a PassiveAggressive, Logistic Regression, Decision Tree, Gradient Boosting, Random Forest and fit the model
5) Testing model with manual entry
