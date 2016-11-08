# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:17:24 2016

@author: g01014411
"""
import argparse
from Featurization import load_bin_vec, vector_from_line, cleantext, removeStopwordswithStemming, score_afinn


parser = argparse.ArgumentParser(description='This Script is used to create the Vector Representation for Words')
parser.add_argument('-af','--afinn_path',help = 'Path of the Afinn word list. Can be downloaded from this URL http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/6010/zip/imm6010.zip', required = False)


requiredNamed = parser.add_argument_group('Required Named Arguments')
requiredNamed.add_argument('-v','--vocab_file', help='Path of Pre-Trained Word Vectors in tab seperated format. Further information about the format is available in the Readme file',required=True)
requiredNamed.add_argument('-t','--text',help = 'Input text for which the Vectors need to be formed',required=True)
args = parser.parse_args()

#User Inputs ----------------------------------------------------------------
vocab_file = args.vocab_file
text = args.text

if args.afinn_path is not None:
    afinn_path = args.afinn_path

def main():
    


    glove_vec, vocab_size = load_bin_vec(vocab_file)
    vocab_size = vocab_size - 1
    
    
    
    text_clean = cleantext(removeStopwordswithStemming(text))
    print("\n\n ------------------------------------------------------------------- ")
    print("\n \t\t The cleaned text is \n")
    print("------------------------------------------------------------------- \n")
    print(text_clean)
    print("\n\n ------------------------------------------------------------------- ")
    line_vector = vector_from_line(text_clean,glove_vec,vocab_size)
    print("\n \t\t The vector representation of the sentence \n\t\t'" + text + "'\n \t\tis \n")
    print("--------------------------------------------------------- \n")
    print(line_vector)
    print("--------------------------------------------------------------------- \n")
    if args.afinn_path is not None:
        print("\n\n ------------------------------------------------------------------- ")
        sentiment = score_afinn(text_clean)
        print("\n \t\t The Afinn Sentiment of the sentence \n\t\t'"+ text + "'\n \t\tis \n" )
        print("--------------------------------------------------------- \n")
        print(sentiment)
        print("--------------------------------------------------------- \n")

if __name__ == '__main__':
    main()
    exit()
