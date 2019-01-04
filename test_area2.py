import tensorflow as tf 
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt 
import skimage.io as io 
import matplotlib.pyplot as plt



file_dir = "C://Users//KG//Desktop//MMAI_894//Project//Images//" #kevin's desktop


images = []
temp = []
labels = []
label_recorder =[]

for root, sub_folders, files in os.walk(file_dir):
    for name in files:
        images.append(os.path.join(root,name)) #get image path
        label_name = root.split('/')[-1]  #split and find label name
        labels = np.append(labels, label_name) #append label name to a list
        label_name = ""
        #print(os.path.join(root, name))
    for name in sub_folders:
        temp.append(os.path.join(root,name))
        label_recorder=np.append(label_recorder,name.split('/')[-1])
############ Lets create a dictionary list with label:int ################
count = 0
label_dic = dict()
for i in label_recorder:
    #print(i)
    label_dic[i]=count
    count +=1  
############ Lets change all the label to numeric based on dictionary ################
labels_copy = np.array(labels)#copy a label version for testing purpose

count = 0
for i in labels_copy:
    #print(i)
    labels_copy[count] = label_dic[i]
    count+=1

#combine images + labels
temp = np.array([images,labels_copy])
temp = temp.transpose()

data = []
trainData =[]
testData =[]
temp_pd = pd.DataFrame(temp)
unique_class = len(set(temp[:,1]))
for i in range(unique_class):
    #print(i)
    data = temp[temp_pd.loc[:,1] == str(i)]
    for i in range(0,int(.8*len(data))):
        trainData.append(data[i]) 
    for j in range(int(.8*len(data)),len(data)):
        testData.append(data[j])
trainData = np.array(trainData)
testData = np.array(testData)
# len(temp)
# len(trainData)
# len(testData)
np.random.shuffle(trainData) 
np.random.shuffle(testData) #randomize the images lists

#split into image and label list
#train split
image_tr = list(trainData[:,0])
label_tr = list(trainData[:,1])
label_tr = list(map(int,label_tr))
#test split
image_test = list(testData[:,0])
label_test = list(testData[:,1])
label_test = list(map(int,label_test))
return image_tr, label_tr,image_test,label_test
