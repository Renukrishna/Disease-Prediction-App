from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import nltk
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
# from google.colab import drive

nltk.download('punkt') 
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

def cleanAndConvertTrainingDataSet(filename):

    #list to contain data
    data = []

    with open(filename) as csvfile:

        rawData = csvfile.readlines()                                               
        for line in rawData:
            # list to contain disease and symptom. Here, the 2nd item contains all symptoms seperated by space                        
            diseaseToSymptom = []

            # string to format all symptoms seprated by space
            string = '' 

            # datasetList is a list of all Symptoms for the disease in its row
            datasetList = line.split(',') 
            for item in datasetList:
                if datasetList.index(item) == 0:
                    diseaseToSymptom.append(item)
                else:
                    string = string + item.rstrip() + ' '
            diseaseToSymptom.append(string)
            
            # here, we get the list of every disease with its possible symptoms, in format [[disease, {{string of symptoms}}] ,   .....]
            data.append(diseaseToSymptom) 
                                            
    return data

def getDiseases(filename):
  
    # This is function which loads all diseases from the file
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        diseases = []
        for row in csvreader:
            diseases.append(row[0])
    return diseases

def trainAndTest(testData):
  '''
        This function takes symptoms as input and predicts the disease
        params: testData(type:string):a string as symptoms seperated by space
        return: prediction as string
  ''' 
  testData = testData
  tokenizedTestData = word_tokenize(testData)

  columns = ['symptom', 'disease',]
  rows = []

  # the file in which our training dataset is available
  filename = "DiseaseSymptomTrainingSetNaiveBayes.csv"

  # cleanAndConvertTrainingDataSet is a function which takes in filename as parameter and formats the data accordingly
  rows = cleanAndConvertTrainingDataSet(filename)
  for i in range(len(rows)):
    rows[i].reverse()

  # get the data into a dataframe 
  trainingData = pd.DataFrame(rows, columns=columns)

  # a dataset which consists of the names of all the diseases
  filename = 'Diseases.csv'

  # getDiseases is a function which loads the names of all the diseases (41 in number)
  diseasesNames = getDiseases(filename)

  allDiseasesDocs = []
  allVec = []
  allX = []
  allTDM = []
  diseaseDocs = []

  for diseaseName in diseasesNames:
    if diseaseName == 'Hypertension ': 
      diseaseDocs = [row['symptom'] for index,row in trainingData.iterrows() if row['disease'] == 'Hypertension ']
      allDiseasesDocs.append(diseaseDocs)
    
    elif diseaseName == 'Diabetes ':
      diseaseDocs = [row['symptom'] for index,row in trainingData.iterrows() if row['disease'] == 'Diabetes ']
      allDiseasesDocs.append(diseaseDocs)
      
    else:
      diseaseDocs = [row['symptom'] for index,row in trainingData.iterrows() if row['disease'] == diseaseName.strip() ] # for disease 1 = stmt and get its rows only
      allDiseasesDocs.append(diseaseDocs)   

  for i in range(len(diseasesNames)):
    vecS = CountVectorizer() 
    allVec.append(vecS)
    XS = vecS.fit_transform(allDiseasesDocs[i])
    allX.append(XS)  # basically center the data and fit and transform using mean and std dev for new values as well
    tdmS = pd.DataFrame(XS.toarray(), columns=vecS.get_feature_names())
    allTDM.append(tdmS)

  allWordList = []
  allCountList = []
  allFreq = []

  for i in range(len(diseasesNames)):
    wordListS = allVec[i].get_feature_names();    # we get the symptom names 
    allWordList.append(wordListS)
    countListS = allX[i].toarray().sum(axis=0)    # we get the frequency in the particular disease class
    allCountList.append(countListS)
    freqS = dict(zip(wordListS,countListS))
    allFreq.append(freqS)  
    
  docs = [row['symptom'] for index,row in trainingData.iterrows()]

  vec = CountVectorizer()
  X = vec.fit_transform(docs)

  totalFeatures = len(vec.get_feature_names()) 

  allTotalCountFeatures = []

  for i in range(len(diseasesNames)):
    totalCountsFeaturesS = allCountList[i].sum(axis=0)
    allTotalCountFeatures.append(totalCountsFeaturesS)

  allProbabilitySymptomWithLS = []
  probabilitySymptomWithLS = []
  probabilitySum = []

  for i in range(len(diseasesNames)):
    for word in tokenizedTestData:
      if word in allFreq[i].keys():
          count = allFreq[i][word]
      else:
          count = 0
      probabilitySymptomWithLS.append((count + 1)/(allTotalCountFeatures[i] + totalFeatures)) #laplace smoothing
    allProbabilitySymptomWithLS.append(dict(zip(tokenizedTestData,probabilitySymptomWithLS)))
    probabilitySymptomWithLS.clear()

  valuesList = []
  finalSum = []
  for i in range(len(allProbabilitySymptomWithLS)):
    valuesList.append(list(allProbabilitySymptomWithLS[i].values()))

  for i in range(len(valuesList)):
    sum = 0
    for j in range(len(valuesList[i])):
      sum+=valuesList[i][j]
    finalSum.append(sum)

  for i in range(len(finalSum)):
    finalSum[i] = finalSum[i]/len(valuesList[i])
  # print(finalSum.index(max(finalSum)))
  return diseasesNames[finalSum.index(max(finalSum))]
# file_dir = os.path.dirname('/Users/rehegde/Documents/ProjectUI/ML-React-App-Template/service')
# sys.path.append(file_dir)
# sys.path.append("..") 

# from . import NaiveBayes


flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Disease Prediction App", 
		  description = "Predict results using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'symptoms': fields.String(required = True, 
				  							   description="Symptoms as text seprated by space", 
    					  				 	   help="Text Field 1 cannot be blank")})

# classifier = joblib.load('classifier.joblib')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 

			formData = request.json
			data = [val for val in formData.values()]
			symptoms_string = str(data[0])
			# symptoms_list = symptoms_string.split()
			try:
				prediction = trainAndTest(symptoms_string)
			except:
				prediction = 'error in prediction'
			# prediction = classifier.predict(data)
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Prediction: " + prediction
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})