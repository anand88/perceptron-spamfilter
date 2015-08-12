import glob
import random
import sys
import os

#peaks at 91.4225 at 20 iterations

#extract vocabulary and number of files
def extractVocAndNumFiles(argPaths, argVocab):
	print("extractVocabulary")
	j = 0
	lNumFiles = -1
	for path in argPaths:
		for lFilename in glob.glob(os.path.join(path, '*.txt')):
			with open(lFilename, encoding='latin-1') as infile:
				for line in infile:
					a = line.strip("\n").split(" ")
					for i in range(len(a)):
						if not a[i] == '' and a[i] not in argVocab.keys():
							argVocab[a[i]] = j
							j = j + 1
			lNumFiles += 1
	return lNumFiles

#extract tokens from documents 
def extractTokensFromDoc(argFilename, argVocab, argGloriousT):
	with open(argFilename, encoding='latin-1') as inFile:
		for line in inFile:
			a = line.strip("\n").split(" ")
			for token in a:
				if token != " " and token in argVocab.keys():
					argGloriousT[argVocab[token]] += 1

#train the perceptron!
def trainP(argPaths, argW, argVocab, argNumFiles):
	lRate = 0.05
	filenames = [glob.glob(os.path.join(argPaths[i], "*.txt")) for i in range(len(argPaths))]
	print("trainP")
	randFile = []
	for k in range(30):
		itr = [0 for i in range(len(filenames))]
		tmpRand = []
		rFIndex = 0
		wrongTrain = 0
		totalTrain = 0
		with open("fileOrder.txt", 'r') as fileOrder:
			for q in range(argNumFiles):
				#FORWARD PROPOGATION
				output = 0
				tokens = [0 for l in range(len(argVocab))]
				k = fileOrder.readline().strip('\n')
				i = int(k)
				extractTokensFromDoc(filenames[i][itr[i]], argVocab, tokens)
				h = 0
				for l in argVocab.keys():
					h += tokens[argVocab[l]]*argW[argVocab[l]]
				h += argW[len(argVocab)]
				if h >0:
					output = 1
				else:
					output = -1
				itr[i] += 1
				#BACKPROPOGATION
				if i == 0:
					i = -1
				if output != i:
					wrongTrain += 1
				for a in argVocab.keys():
					argW[argVocab[a]] += lRate*(i-output)*tokens[argVocab[a]]
				argW[len(argVocab)] += lRate*(i-output)
				rFIndex += 1
				totalTrain += 1
			print("TRAINING wrong : ", wrongTrain, " total : ", totalTrain, "accuracy : ", 100*(totalTrain - wrongTrain)/totalTrain)

#test the perceptron
def testP(argPathsTest, argW, argVocab):
	wrong = 0
	total = 0
	for j in argPathsTest:
		for filename in glob.glob(os.path.join(j, "*.txt")):
			h = 0
			token = [0 for i in range(len(argVocab))]
			extractTokensFromDoc(filename, argVocab, token)
			for l in argVocab.keys():
				h += token[argVocab[l]]*argW[argVocab[l]]
			h += argW[len(argVocab)]
			if h >0:
				output = 1
			else:
				output = 0
			print("output : ", output, " path : ", j, " filename : ", filename) 
			if output != argPathsTest.index(j):
				wrong +=1
			total += 1
	print("WRONG : ", wrong, "TOTAL : ", total, "ACCURACY : ", (100*(total-wrong)/total))

def main():     
	i = 1
	paths = []
	pathsTest = []
	while sys.argv[i] != 'test':
		paths.append(sys.argv[i])
		print('train', sys.argv[i])
		i += 1
	i += 1
	while i < len(sys.argv):
		pathsTest.append(sys.argv[i])
		print('test', sys.argv[i])
		i += 1
	print("pathsTest ", pathsTest)
	print("paths ", paths)
	vocab = {}
	numFiles = extractVocAndNumFiles(paths, vocab)
	print("vocab : ", vocab)
	weights = [0 for i in range(len(vocab)+1)]
	trainP(paths, weights, vocab, numFiles)
	print("weights: ", weights)
	testP(pathsTest, weights, vocab)

if __name__ == "__main__":
    main()
