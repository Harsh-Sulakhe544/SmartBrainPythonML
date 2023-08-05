import os # cuz we want all 3 images 
from imageai.Classification import ImageClassification # to install modules to detect image ,IMPORT DATA

execution_path=os.getcwd() # to grab the current-working-directory

#  to use the model : MobileNetV2
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth"))
prediction.loadModel()

# to make the predictions : 
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)