from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def classify(img_file):
    img_name=img_file
    test_image = image.load_img(img_name, target_size=(256,256), color_mode='grayscale')

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    results = loaded_model.predict(test_image)
    arr=np.array(results[0])
    print("Array of results",arr)

    maxx=np.amax(arr)
    max_prob=arr.argmax(axis=0)
    max_prob=max_prob + 1
    classes=['NONE','ONE','TWO','THREE','FOUR','FIVE']
    results=classes[max_prob-1]
    print("Ing_name",img_name,"Result",results)

import os 
path = 'E:\Data-----Science\DS AI DL ML Project for 30 Days\Hand Gesture recognition\check'

files=[]
#r= root , d = directories , f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))

for f in files:
    classify(f)
    print("\n")
