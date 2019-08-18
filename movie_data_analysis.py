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

class Visualize:
    def __init__(self):
        self.df1 = pd.read_csv('./movie_name_char_mentions_centrality.csv')
        self.df2 = pd.read_csv('./movie_emotion_year.csv')
        self.df3 = pd.read_csv('./movie_singer_count.csv')
        self.df4 = pd.read_csv('./movie_plot.csv')
        self.df5 = pd.read_csv('./movie_all.csv')



    def lead_role(self, q):
        col = self.df1[self.df1['movie']==q]
        if(col.empty):
            print("The movie ", q, " is not found in the database. Cannot find the lead role", sep="")
            return
        ser = col['name']
        result = 'actor' in ser.values
        if(result):
            print("The lead role is 'actor'")
            print("The type of role played is: ", col[col['name'].values=='actor']['character'])

        else:
            col = col.sort_values(by=['count'], ascending=False)
            ser = col['name']
            nam = ser.values[0]
            ind = ser[ser==nam].index[0]
            print("The lead role is:", nam)
            print("The type of role played is: ", self.df1[self.df1['index']==ind]['character'].values[0])


    def characters(self, q):
        col = self.df1[self.df1['movie']==q]
        if(col.empty):
            print("The movie ", q, " is not found in the database. Cannot find the characters", sep="")
            return
        ser = col['name']
        print("The characters in the movies", q, "include:")
        print(col[['name', 'character']])


    def character(self, q, m):
        col = self.df1[self.df1['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the character", sep="")
            return
        ser = col['name']
        nam = "NULL"
        try:
            nam = ser[ser==q].values[0]
        except IndexError as e:
            print("The character ", q, " is not found in the database.Cannot find the character.", sep="")
            return
        ind = ser[ser==nam].index[0]
        print("The role is:", nam)
        print("The type of role played is: ", self.df1[self.df1['index']==ind]['character'].values[0])



    def plot(self, m):
        pd.set_option('display.max_colwidth', -1)
        col = self.df4[self.df4['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the plot of the movie", sep="")
            return
        print("The plot of the film goes like: ")
        print(col['plot'])

    def appearances(self, c, m):
        col = self.df1[self.df1['movie'] == m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find appearances.", sep="")
            return
        ser = col['name']
        try:
            nam = ser[ser==c].values[0]
        except IndexError as e:
            print("The character ", c, " is not found in the database.Cannot find the appearances.", sep="")
            return
        ind = ser[ser==nam].index[0]
        print("The role is:", nam)
        print("The number of appearances are: ", self.df1[self.df1['index']==ind]['count'].values[0])
        print("The average centrality is: ", self.df1[self.df1['index']==ind]['average centrality'].values[0])
        print("The total centrality is: ", self.df1[self.df1['index']==ind]['total centrality'].values[0])
        

def genre(self, m):
        col = self.df2[self.df2['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the genre.", sep="")
            return
        se = col['emotion'].value_counts()
        gen = se[se == max(se.values)].index[0]

        def genre(self, m):
            col = self.df2[self.df2['movie']==m]
            if(col.empty):
                print("The movie ", m, " is not found in the database. Cannot find the genre.", sep="")
                return
            se = col['emotion'].value_counts()
            gen = se[se == max(se.values)].index[0]

            
        if(gen=="happy"):
            genre = "Family-Entertainer"
        elif(gen == "neutral"):
            genre = "Drama"
        elif(gen == "sad"):
            genre = "Melo-Drama"
        elif(gen == "angry"):
            genre = "Action"
        elif(gen == "fear"):
            genre = "Horror"
        elif(gen=="suprise"):
            genre = "Suspence Thriller"
        elif(gen=="disgust"):
            genre = "Crime-Thriller"
        print("GENRE:")
        print("The movie ", m, " is a ", genre, " genre film.", sep="")
        











            
        def image_movie(self, arr):
        for i in range(len(arr)):
            if arr[i] in self.df5.iloc[:, -1].values:
                m = arr[i]
                print(arr[i])
                self.lead_role(m)
                self.characters(m)
                self.plot(m)
                self.year(m)
                self.songs(m)
                self.average_emotion(m, 0)
                self.length_of_movie(m)
                return
        print("Could not find the movie in the dataset. Try another image.")

        
        
# input using NLP
ext = 0         # checking for exit condition  
while ext!=1:
    obj = Visualize()
    print("\n")
    choice = int(input("Enter your choice:\n1.Image Input\n2.Text Input\n3.Exit\n"))
    if choice == 1:
        ob = ImageProcessing()
        tensor = ob.image_process()
        obj.image_movie(tensor)
        continue
    elif choice == 2:
        print("Queries can be framed using the following to get optimum results:")
        print("1.characters\n2.plot\n3.genre\n4.attitude\n5.appearances\n6.year\n7.songs\n8.length\n9.variation\n10.predict\n11.emotion\n12.role\n13.exit\n14.movie\n15.emotions\n16.character\n")
        ob = NLP()
        ob.input_query()
        tensor = ob.processing()
    elif choice == 3:
        print("Interupt Process")
        break;
    else:
        print("Invalid Input")
        continue
    count = tensor[0]
    for i in range(1, tensor[0]+1):
        
        if tensor[i]=="role":
            obj.lead_role(tensor[i + count])
            print("\n")
            
        elif tensor[i]=="characters":
            obj.characters(tensor[i+count])
            print("\n")
        
        elif tensor[i]=="attitude":
            obj.character(tensor[i+count], tensor[i+count+1])
            count += 1
            print("\n")
        
        elif tensor[i]=="plot":
            obj.plot(tensor[i+count])
            print("\n")

        elif tensor[i]=="appearances":
            obj.appearances(tensor[i+count], tensor[i+count+1])
            count += 1
            print("\n")
            
        elif tensor[i]=="year":
            obj.year(tensor[i+count])
            print("\n")
            
        elif tensor[i]=="songs":
            obj.songs(tensor[i+count])
            print("\n")

        elif tensor[i]=="emotion":
            try:
                if ("average" in ob.temp):
                    obj.average_emotion(tensor[i+count], 0)
                    print("\n")
                    break
                elif ("minor" in ob.temp):
                    obj.average_emotion(tensor[i+count], 1)
                    print("\n")
                    break
                elif ("major" in ob.temp):
                    obj.average_emotion(tensor[i+count], 2)
                    print("\n")
                    break
                elif("predict" in ob.temp):
                    obj.predict()
                    print("\n")
                    break
            except IndexError as e:
                print("Not enough parameters")
                break
        
        elif tensor[i]=="genre":
            obj.genre(tensor[i+count])
            print("\n")
            
        elif tensor[i]=="length":
            obj.length_of_movie(tensor[i+count])
            print("\n")

        elif tensor[i]=="variation":
            obj.trends(True)
            print("\n")
            
            
        elif tensor[i]=="exit":
            print("Process Interupt")
            ext = 1
        else:
            print("Query does not contain enough parameters.")

    
        
