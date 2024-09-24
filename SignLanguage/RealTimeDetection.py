from keras.models import model_from_json ; import numpy as np ; import cv2 ; from keras.preprocessing.image import load_img

file = open("model.json" , "r")
model_json = file.read()
file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")


Labels = ["A" , "B" , "C" , "D" , 'E' , 'F' , 'G' , 'H' , 'I' , 'K' , 'L' , 'M' , 'N' ,
          'O' , 'P' , 'Q' , 'R' , 'S' , 'T' , 'U' , 'V' , 'W' , 'X' , 'Y']

def Extract(image):
    image = np.array(image)
    image = cv2.resize(image , (28 , 28))
    im = []
    im.append(image)
    im = np.array(im)
    im.reshape(-1 , 28 , 28 , 1)
    im = im/255.0
    return im

cap = cv2.VideoCapture(0)
c = []
while True:
    ret , frame = cap.read()
    cv2.rectangle(frame , (0 , 40) , (300 , 300) , (255 , 255 , 255) , 2)
    CropFrame = frame[40:300 , 0:300]
    CropFrame = cv2.cvtColor(CropFrame , cv2.COLOR_BGR2GRAY)
    CropFrame = Extract(CropFrame)
    pred = model.predict(CropFrame)
    prediction_label = Labels[pred.argmax()]
    cv2.rectangle(frame , (0,0) , (300 , 40) , (0 , 165 , 255) , -1)
    if max(pred[0]) < 0.1:
        cv2.putText(frame , "None" , (10 , 30) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 1 , (0,0,255),2 , cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred)*100)
        cv2.putText(frame , f"{prediction_label} {accu}%", (10 , 30) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 1 , (0,0,255))

    cv2.imshow("Model" , frame)
    if cv2.waitKey(1) == ord("q"):
        break
print(CropFrame.shape)
cap.release()
cv2.destroyAllWindows()