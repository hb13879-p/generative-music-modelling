from keras import backend as K

def chord_accuracy(y_true,y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
