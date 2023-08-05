from imageai.Classification import ImageClassification # to install modules to detect image ,IMPORT DATA
import os # cuz we want all 3 images 

exec_path = os.getcwd() # to grab the current-working-directory

#  to use the model : Resnet
prediction = ImageClassification()
# SqueezeNet model also no longer exists, now the fastest is MobileNetV2
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(exec_path, 'resnet50-19c8e357.pth'))
prediction.loadModel()

# to make the predictions : result_count ==> no of parameters to predict (comparing -- skin , height , background , water-level , size , shape)
print("\n\n my predictions for giraffee are : ")
predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'giraffe.jpg'), result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')

print("\n\n my predictions for godzilla are : ")
predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'godzilla.jpg'), result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')

print("\n\n my predictions for house are : ")
predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'house.jpg'), result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')
    
    

