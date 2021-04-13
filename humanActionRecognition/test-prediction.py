import keras
from utils import compressed_pickle, decompress_pickle, plotLearningCurve
from ourGenerator import OurGenerator
from keras.applications.resnet50 import ResNet50
import numpy as np


model = keras.models.load_model('models/ours_7_10epochs_model', compile = True)

labels = decompress_pickle('old/labels.pickle.pbz2')
partition = decompress_pickle('old/partition.pickle.pbz2')

resnet = ResNet50(include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(240, 320, 3),
                pooling=None
                )

correct_count = 0
incorrect_count = 0

for ID in partition['test']:
    x = np.array(decompress_pickle('old/new_inputs/' + ID + '.pickle.pbz2'))
    x = keras.applications.resnet.preprocess_input(x)
    x = resnet.predict(x)
    x = x.reshape(32, -1, 2048)
    prediction = model.predict(x)[0]
    prediction = np.argmax(prediction)
    label = labels[ID]
    if label == prediction:
        correct_count += 1
        print('correct')
        print(ID)
    else:
        incorrect_count += 1
        print('incorrect')
        print(ID)
        print(label)
        print(prediction)

print('accuracy')
accuracy = float(correct_count) / float(correct_count + incorrect_count)
print(accuracy)
