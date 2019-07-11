from keras.models import load_model

mod = load_model('./drive/My Drive/Colab Notebooks/SigNet/colab_08072019.h5',custom_objects={'contrastive_loss':contrastive_loss})

im_1 = cv2.imread("path/to/image")
im_2 = cv2.imread("path/to/image")  
im_1 = cv2.resize(im_1,(220,155))
im_2 = cv2.resize(im_2,(220,155))
im_1 = cv2.bitwise_not(im_1)
im_2 = cv2.bitwise_not(im_2)
im_1 = im_1/255
im_2 = im_2/255
im_1 = np.expand_dims(im_1,axis=0)
im_2 = np.expand_dims(im_2,axis=0)

y_pred = mod.predict([im_1,im_2])
print(y_pred)