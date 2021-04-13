# This code is taken from a Stack Over Flow answer!

import json, codecs
import keras
import os


history_filename = 'history-checkpoints/ours_3.json'

def saveHist(path, history):
    with codecs.open(path, 'w+', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)

def loadHist(path):
    n = {} # set history to empty
    if os.path.exists(path): # reload history if it exists
        with codecs.open(path, 'r', encoding='utf-8') as f:
            n = json.loads(f.read())
    return n

def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest

class LossHistory(keras.callbacks.Callback):

    # https://stackoverflow.com/a/53653154/852795
    def on_epoch_end(self, epoch, logs = None):
        new_history = {}
        for k, v in logs.items(): # compile new history from logs
            new_history[k] = [v] # convert values into lists
        current_history = loadHist(history_filename) # load history from current training
        current_history = appendHist(current_history, new_history) # append the logs
        saveHist(history_filename, current_history) # save history from current training
