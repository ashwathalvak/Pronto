from __future__ import print_function

import numpy as np

import re
try:
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize
except ImportError:
    def downloadNLTKDependancy():
        """
        Script to download the required NLTK dependecies
        """
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
            
    downloadNLTKDependancy()
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize

    
    
        
  

    
from itertools import groupby
import math

# Functions: General ----------------------------------------------------------

#Function is used to read the vocabulary file. Has been tested on the GloVe vocabulary file
#vocab_file - Path to the text file, containing the vectors. Only space/tab seperated file supported

def load_bin_vec(vocab_file):
    glove_vecs = {}
    with open(vocab_file, "rb") as file_connection:
        for row,line in enumerate(file_connection):
            line_vec=line.split()
            glove_vecs[line_vec[0]]=map(float,line_vec[1:])
    
    return glove_vecs, len(line_vec)



#Function is used to return the numeric vector for the line of text passed to it. Arguments that need to be supplied are
    #line - String - Text input
    #glove_vec - dict - vocabulary of word vectors
    #vocab_size - int - Number of features that aer generated from the vocabulary file

def vector_from_line(line, glove_vec, vocab_size):
    vec = np.zeros(vocab_size)
    for word in line.split():
        try:
            vec = np.add(vec , glove_vec[word])
        except KeyError:
            pass
    return vec


#Converts emoticons to text, identifies emoticons in 4 categories blanks "" (others), "joking", "sad" and "happy"
# Cleans up URL's
#Cleans up whitespaces to a single blank space
# Removes the hash from hashtags
# first occurence of @<text> is replaced as AT_USER
# Removes ascii character
#Duplicate letters beyond 2 letters are removed
def cleantext(text):
    NormalEyes = r'[:]'
    Wink = r'[;]'
    NoseArea = r'[-\'`\\]*'
    HappyMouths = r'[dD)\]]'
    SadMouths = r'[\(\[]'
    Tongue = r'[pP]'
    OtherMouths = r'[oO\/\\|]'  # remove forward slash if http://'s aren't cleaned
    Happy_RE =  re.compile( NormalEyes + NoseArea + HappyMouths) # :d,:D,:],:),:-D,:-) ,:-],:-d, :'), :'],:'D,:'d, :`), :`],:`D,:`d, :\), :\], :\D , :\d
    Sad_RE = re.compile(NormalEyes + NoseArea + SadMouths) # :(, :[, :-( ,:-[, :'(, :'[, :`(, :`[, :\(, :\[
    Wink_RE = re.compile(Wink + NoseArea + HappyMouths) # ;D,;],;),;-D,;-) ,;-],;-D, ;'), ;'],;'D, ;`), ;`],;`D, ;\), ;\], ;\D
    Tongue_RE = re.compile(NormalEyes + NoseArea + Tongue) # :p, :P, :-p ,:-P, :'p, :'P, :`p, :`P, :\p, :\P
    Other_RE = re.compile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths ) # :o, :O, :/, :\, :-o, :-O, :-/, :-\, :'o, :'O, :'/, :'\, :`o, :`O, :`/, :`\, :\o, :\O, :\/, :\\, ;o, ;O, ;/, ;\, ;-o, ;-O, ;-/, ;-\, ;'o, ;'O, ;'/, ;'\, ;`o, ;`O, ;`/, ;`\, ;\o, :\O, ;\/, ;\\
    
    text = text.decode('utf-8')
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','|URL|',text)   #replaces urls with |URL|
    text = re.sub('@[^\s]+','|AT_USER|',text)  # occurence of @<anywordwithoutspace> with AT_USER, ensures @ symbol is the first character
    text = re.sub('[\s]+', ' ', text) #[ \t\n\r\f\v], replaces with ' '
    text = re.sub(r'\#([^\s]+)', r'\1', text)  #removes the hash when it is the first character
    text = re.sub(r'[\x90-\xff]', '', text) #Remove ascii decoding errors
    text = text.replace('-',' ')
    text = text.strip('\'"')
    
    text = text.replace('\"','')
    
    text = text.replace("\`",'')
    
    text = str(''.join(''.join(s)[:2] for _, s in groupby(text)))  #Removes duplicates beyond 2 letters,  wayyyyyyy will be converted to wayy
    text = Other_RE.sub("",text)
    text = Tongue_RE.sub("joking",text)
    text = Wink_RE.sub("joking",text)
    text = Sad_RE.sub("sad",text)
    text = Happy_RE.sub("happy",text)
    text = text.lower()
    return text

#converts text to lowercase
#removal of stopwords
#Applies snowball stemmer

def removeStopwordswithStemming(text):
    # twts = dat.tweet.values
    # lower case
    text = text.lower()
    # remove stop-words
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer("english")
    
    text = str(' '.join([stemmer.stem(word.replace("'s",''))
                      for word in word_tokenize(text)
                      if word not in stop_words])
            )

    return text

#Score a simple Afinn sentiment model
def score_afinn(text):
    """
        Returns a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative valence.
        """
    afinn = dict(map(lambda (w, s): (w, int(s)), [
                            ws.strip().split('\t') for ws in open(afinn_path) ]))
        
    words = text.split()
    sentiments = map(lambda word: afinn.get(word, 0), words)
    if sentiments:
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
    else:
        sentiment = 'neutral'
                                                                                  
    if sentiment > 0.1:
        sentiment = 'positive'
    elif sentiment > -0.1:
        sentiment = 'neutral'
    elif sentiment <= -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
            
    return sentiment

#Calling function for the loading the vectors and finding the vector for the text input
