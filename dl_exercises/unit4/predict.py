from keras.models import load_model
from dataset import load_dataset

def predict_imgs(X_test):
    # Load model
    model = load_model('cnn_model.h5')

    # Load images
    img = X_test[0].reshape(1,32,32,3)

    # Predict
    y_pred = model.predict_classes(img)

    return y_pred

X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
print(predict_imgs(X_test))