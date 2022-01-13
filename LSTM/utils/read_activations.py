from keras import backend as K

# with a Sequential model

def get_activations(model,layer_no,evaluation_data):
    get_layer_output = K.function([model.layers[0].input],[model.layers[layer_no].output])
    return get_layer_output([evaluation_data])[0]
