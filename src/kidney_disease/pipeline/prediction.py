import numpy as np
import tensorflow as tf
import os



class PredictionPipeline:
    def __init__(self):
        pass


    
    def model_predict(img_path,model):
        
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
        x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
        x=x/255
        x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

        preds = model.predict(x)
        preds=np.argmax(preds, axis=1)
        if preds==0:
            prediction = 'Tumor'
        elif preds==1:
            prediction = 'Normal'
   
        return prediction