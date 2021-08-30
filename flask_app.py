from flask import Flask, render_template
from flask import request
from flask import redirect, url_for
import os
from cv2 import cv2
import pickle
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io

from sklearn.decomposition import PCA
from PIL import Image


app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

## -------------------- Load Models -------------------
model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_sgd.pickle')
scaler_path = os.path.join(MODEL_PATH,'dsa_scaler.pickle')
haar_path=os.path.join(MODEL_PATH,'haarcascade_frontalface_default.xml')
mean_path=os.path.join(MODEL_PATH,'mean_preprocess.pickle')
model_svm_path=os.path.join(MODEL_PATH,'model_svm.pickle')
model_pca_path=os.path.join(MODEL_PATH,'pca_50.pickle')


haar=cv2.CascadeClassifier(haar_path)
mean=pickle.load(open(mean_path,'rb'))
model_svm=pickle.load(open(model_svm_path,'rb'))
model_pca=pickle.load(open(model_pca_path,'rb'))
model_sgd = pickle.load(open(model_sgd_path,'rb'))
scaler = pickle.load(open(scaler_path,'rb'))


@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 aala re . Page Not Found. Please go the home page and try again"
    return render_template("error.html",message=message) # page not found

@app.errorhandler(405)
def error405(error):
    message = 'Error 405, Method Not Found'
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message='INTERNAL ERROR 500, Error occurs in the program'
    return render_template("error.html",message=message)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/faceapp/')
def faceapp():
    return render_template("faceapp.html")

@app.route('/faceapp/gender/',methods=['GET','POST'])
def gender():
    if request.method=='POST':
        f=request.files['image']
        filename=f.filename
        path=os.path.join(UPLOAD_PATH,filename)
        f.save(path)
        w=get_width(path)
        # Predictions (Passing to pipeline model)
        pipeline_model_faceapp(path,filename,color='bgr')
        return render_template("gender.html",fileupload=True,img=filename,w=w)

    return render_template("gender.html",fileupload=False,img="veoim.jpg", w='250')



@app.route('/imageclassifier/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename 
        print('The filename that has been uploaded =',filename)
        # know the extension of filename
        # all only .jpg, .png, .jpeg, PNG
        ext = filename.split('.')[-1]
        print('The extension of the filename =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            # send to pipeline model
            results = pipeline_model_image(path_save,scaler,model_sgd)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)


        else:
            print('Use only the extension with .jpg, .png, .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)
            
    else:
        return render_template('upload.html',fileupload=False,extension=False)

@app.route('/about/')
def about():
    return render_template('about.html')

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ =img.shape 
    aspect = h/w
    given_width = 300
    height = given_width*aspect
    return height

def get_width(path):
    img=Image.open(path)
    size=img.size
    aspect_ratio= size[0]/size[1]  # width/height
    w=250*aspect_ratio
    return int(w)

def pipeline_model_image(path,scaler_transform,model_sgd):
    # pipeline model
    image = skimage.io.imread(path)
    # transform image into 80 x 80
    image_resize = skimage.transform.resize(image,(80,80))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    # rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    # hog feature
    feature_vector = skimage.feature.hog(gray,
                                  orientations=10,
                                  pixels_per_cell=(8,8),cells_per_block=(2,2))
    # scaling
    
    scalex = scaler_transform.transform(feature_vector.reshape(1,-1))
    result = model_sgd.predict(scalex)
    # decision function # confidence
    decision_value = model_sgd.decision_function(scalex).flatten()
    labels = model_sgd.classes_
    # probability
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)
    
    # top 5
    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]
    # put in dictornary
    top_dict = dict()
    for key,val in zip(top_labels,top_prob):
        top_dict.update({key:np.round(val,3)})
    
    return top_dict






def pipeline_model_faceapp(path,filename,color='rgb'):
    # read image in cv2
    
    gender_pre={0:'Male',1:'Female'}
    font=cv2.FONT_HERSHEY_SIMPLEX   
    img=cv2.imread(path)
    if color=='bgr':
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    faces=haar.detectMultiScale(gray,1.5,3)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #Cropping
        cp=gray[y:y+h,x:x+w]
        #Normalisation
        cp=cp/255.0
        #Resizing image into (100,100)
        if cp.shape[1]>100:
            cp_resize=cv2.resize(cp,(100,100),cv2.INTER_AREA)
        else:
            cp_resize=cv2.resize(cp,(100,100),cv2.INTER_CUBIC)
        # Flattening 1x 10000

        cp_reshape=cp_resize.reshape(1,10000) # 1x 10000
        #plt.imshow(cp_reshape)
        #subtracting with mean
        cp_mean=cp_reshape-mean # saver is mean
        # Eigen image
        eigen_image=model_pca.transform(cp_mean)

        # Pass to ml Model(Support Vector Machine)
        results=model_svm.predict_proba(eigen_image)[0]

        # prediction
        predict=results.argmax() # 0 or 1
        score=results[predict]

        #sss
        text= "%s : %0.2f"%(gender_pre[predict],score)
        cv2.putText(img,text,(x,y),font,1,(0,255,255),3)
    pred_path=os.path.join(BASE_PATH,'static/prediction')
    pred_final=os.path.join(pred_path,filename)
    cv2.imwrite(pred_final,img)




if __name__ == "__main__":
    app.run(debug=True) 