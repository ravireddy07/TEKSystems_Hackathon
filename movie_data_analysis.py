import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ImageProcessing:

    def image_process(self):
        import pytesseract
        from PIL import Image
        import cv2
        import os
        path=r'c:\data_analysis'

        img_path=input("Enter the complete path of the image file")
        img=Image.open(img_path)
        if not os.path.exists(path):
            os.makedirs(path)
        print("Hold until we process your Image.....")
        #convert the image type into numpy array for processing
        image_data = np.asarray(img)
        #denoising the image
        dst = cv2.fastNlMeansDenoisingColored(image_data,None,10,10,7,21)
        #saving the denoised image
        cv2.imwrite(r'c:\data_analysis\deionise.png', dst)
        #to convert tesseract_cmd file to tesseract.exe
        pytesseract.pytesseract.tesseract_cmd = "H:/TEKResults/Tesseract 4.0.0/tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = 'H:/TEKResults/Tesseract-OCR/tesseract.exe'
        #converting the image into grayscale
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        #saving the grayscale image
        cv2.imwrite(r'H:/TEKResults/enhancedGrayscaleLineCapture.png', gray)
        #to increase the threshold of the image
        th1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,11,2)
        ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blue, green, red = cv2.split(image_data)
        blue_edges = cv2.Canny(blue, 100, 10)
        green_edges = cv2.Canny(green, 100, 10)
        red_edges = cv2.Canny(red, 100, 10)
        edges = blue_edges | green_edges | red_edges
        #saving enhanced gray scale threshold,grayscaled images
        cv2.imwrite(r'H:/TEKResults/enhancedGrayscaleThresholdLineCapture.png', th2)
        cv2.imwrite(r'H:/TEKResults/bluegreenred.png', edges)
        img2=Image.open(r'H:/TEKResults/enhancedGrayscaleThresholdLineCapture.png')
        img1=Image.open(r'H:/TEKResults/bluegreenred.png')
        images=Image.open(r'H:/TEKResults/deionise.png')
        #extract text from denoised image
        result=pytesseract.image_to_string(images,lang='eng')
        output_temp=result.split()
        for i in range(len(output_temp)):
            output_temp[i]=output_temp[i].lower()
        output_vectors=[]
        return output_temp


class NLP:

    def __init__(self):
        self.output_vectors=[]
        self.input_text_vectors=[]
        self.constraints_vectors=[]
        self.keywords_vectors=[]
        self.output=[]
        self.num_words=1000
        self.number_of_constraints=0
        ck = pd.read_csv(r"C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\const_key.csv")  #ck.iloc[:, 0]
        self.keywords=ck.iloc[0:3, 1].tolist()
        self.constraints = (ck.iloc[:, 0]).tolist()
        self.input_text = ""
        self.temp =[]
        self.key_count = 0

    def input_query(self):
        self.input_text=input("Enter Your Query:\n")

    def processing(self):
        self.temp=(self.input_text).split(" ")
        l1=self.temp+self.constraints+self.keywords
        l = [l1]
        tokenizer=Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(l)
        token_outputs=tokenizer.word_index
        for i in range(len(self.temp)):
            self.input_text_vectors.append(token_outputs[self.temp[i]])
        for j in range(len(self.constraints)):
            self.constraints_vectors.append(token_outputs[self.constraints[j]])
        for k in range(len(self.keywords)):
            self.keywords_vectors.append(token_outputs[self.keywords[k]])
        for m in range(len(self.input_text_vectors)):
            for n in range(len(self.constraints_vectors)):
                if(self.input_text_vectors[m]==self.constraints_vectors[n]):
                    self.output_vectors.append(self.input_text_vectors[m])
                    self.number_of_constraints+=1
        for o in range(len(self.input_text_vectors)):
            for p in range(len(self.keywords_vectors)):
                if(self.input_text_vectors[o]==self.keywords_vectors[p]):#must handle array index out of bound error and print query incomplete
                    try:
                        self.key_count += 1
                        self.output_vectors.append(self.input_text_vectors[o+1])
                    except IndexError as e:
                        print("Query does not contain enough parameters.")
                        return self.processing()
        self.output.append(self.number_of_constraints)
        for q in range(len(self.output_vectors)):
            for value,vectors in token_outputs.items():
                if (self.output_vectors[q]==vectors):
                    self.output.append(value)
        if 'predict' in self.output:
            return self.output
        if self.number_of_constraints <= self.key_count:
            return self.output
        else:
            print("Not enough keywords")
            self.input_query()

            
def predict(self):
        df_area = self.trends(False)
        print(df_area)
        # Data-Preprocessing
        z = pd.read_csv('./trend_emotion.csv')
        X = z.iloc[:, :-1]
        y = z.iloc[:, -1]
