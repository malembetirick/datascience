#-*-coding:utf-8-*

import urllib2 as lib
from nltk import bigrams
from random import randint

def wordListSum(wordList):
	sum = 0
	#retourner une liste de tuples( paire (cle, valeur)) stockees dans un dico, la valeur correspond à son nombre d'occurences
	for word, value in wordList.items():
		sum += value
	return sum
#retourne aleatoirement les mots pondérés les plus probable par rapport à leurs nombres d'occurence
def retrieveRandomWord(wordList):

	randIndex = randint(1, wordListSum(wordList))
	for word, value in wordList.items():
		randIndex -= value
		if randIndex <=0:
			return word

def buildWordDict(text):

	words = text
	#filrer les mots vides
	words = [word for word in words if word !=""]

	wordDict = {}
	for i in range(1, len(words)):
		if words[i-1] not in wordDict:
		#creer un nouveau dico pour les mots
			wordDict[words[i-1]] = {}
		if words[i] not in wordDict[words[i-1]]:
			wordDict[words[i-1]][words[i]]=0
		wordDict[words[i-1]][words[i]] +=1
	return wordDict

text = str(lib.urlopen("http://www.lifl.fr/~pietquin/teaching/sentences.txt").read())

bigram = list(bigrams(text.split()))
wordDict = buildWordDict(bigram)
#print (wordDict)
#generer une chaine de markov d'une longueur de 100 d'ordre 1
length = 100
chain ="le but de la"
currentWord = "vie est"
for i in range(0, length):
	chain += currentWord+" "
	currentWord = retrieveRandomWord(wordDict[currentWord])

print(chain)
