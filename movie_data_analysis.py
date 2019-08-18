import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Ravi Contribution
class ImageProcessing:
    def image_process(self):
        import pytesseract
        from PIL import Image
        import cv2
        import os
        path=r'c:\data_analysis'
        img_path=input("Enter the complete path of the image file") #Loading Image to process
        img=Image.open(img_path)

        if not os.path.exists(path):
            os.makedirs(path)
        print("Hold until we process your Image.....")
        image_data = np.asarray(img)

        dst = cv2.fastNlMeansDenoisingColored(image_data,None,10,10,7,21)
        cv2.imwrite(r'c:\data_analysis\deionise.png', dst)
        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract 4.0.0/tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(r'c:\data_analysis\enhancedGrayscaleLineCapture.png', gray)
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

        cv2.imwrite(r'c:\data_analysis\enhancedGrayscaleThresholdLineCapture.png', th2)
        cv2.imwrite(r'c:\data_analysis\bluegreenred.png', edges)
        img2=Image.open(r'c:\data_analysis\enhancedGrayscaleThresholdLineCapture.png')
        img1=Image.open(r'c:\data_analysis\bluegreenred.png')
        images=Image.open(r'c:\data_analysis\deionise.png')

        result=pytesseract.image_to_string(images,lang='eng')
        output_temp=result.split()

        for i in range(len(output_temp)):
            output_temp[i]=output_temp[i].lower()
        output_vectors=[]
        return output_temp
#Ravi Contribution Ended


#Sasank Contribution
class NLP:
    def __init__(self):
        self.output_vectors=[]
        self.input_text_vectors=[]
        self.constraints_vectors=[]
        self.keywords_vectors=[]
        self.output=[]
        self.num_words=1000
        self.number_of_constraints=0
        ck = pd.read_csv(r"C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\const_key.csv")
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
                if(self.input_text_vectors[o]==self.keywords_vectors[p]):
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
#Sasank Contribution Ended


#Prem Contribution
class Visualize:
    def __init__(self):
        self.df1 = pd.read_csv(r'C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\movie_name_char_mentions_centrality.csv')
        self.df2 = pd.read_csv(r'C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\movie_emotion_year.csv')
        self.df3 = pd.read_csv(r'C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\movie_singer_count.csv')
        self.df4 = pd.read_csv(r'C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\movie_plot.csv')
        self.df5 = pd.read_csv(r'C:\Users\Feroz\Downloads\Compressed\mouna\movie_data_analysis-master\Movie Analysis\movie_all.csv')

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

    def year(self, m):
        col = self.df2[self.df2['movie'] == m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the year of release.", sep="")
            return
        print("The movie", m, "released in the year",col['year'].values[0])

    def songs(self, m):
        col = self.df3[self.df3['movie'] == m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the songs data.", sep="")
            return
        singers = col['singer_name'].values.tolist()
        print("The movie", m, "has", col['song_count'].sum(), "songs.\n")
        print("And the singers are:\n", "\n ".join(singers))

    def average_emotion(self, m, n):
        col = self.df2[self.df2['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find average emotion.", sep="")
            return
        se = col['emotion'].value_counts()
        if(n==0):
            fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))
            recipe = se.index
            data = se.values
            wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
                      bbox=bbox_props, zorder=0, va="center")
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
            ax.set_title("Average Emotion")

            plt.show()

        maxi = max(se.values)
        mini = min(se.values)
        max_per = (maxi/sum(se.values))*100
        min_per = (mini/sum(se.values))*100
        if(n==2):
            print('\nThe most expressed emotion in the film is "',se[se == maxi].index[0],'"'," and constitutes to ", max_per,"%",sep="")
        if(n==1):
            print('\nThe least expressed emotion in the film is "',se[se == mini].index[0],'"'," and constitutes to ", min_per,"%", sep="")


        if(n==0):
            self.create_wordcloud(col)  #WordCloud Created

        if(n==0):
            self.genre(m)  #Genre of the film

    def create_wordcloud(self, q):
        from wordcloud import WordCloud, STOPWORDS
        print("\n\nThe wordcloud created for the emotions of the data in the film:\n")
        comment_words = ' '
        stopwords = set(STOPWORDS)

        for val in q:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            for words in tokens:
                comment_words = comment_words + words + ' '


        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 10).generate(' '.join(q['emotion']))

        plt.figure(figsize = (4, 4), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
        print("Note: The the size of the word increases with higher expressed emotion.")

    def genre(self, m):
        col = self.df2[self.df2['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the genre.", sep="")
            return
        se = col['emotion'].value_counts()
        gen = se[se == max(se.values)].index[0]

#fuzzying the output
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

#Results Length of the Movie
    def length_of_movie(self, m):
        col = self.df1[self.df1['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the length of the movie", sep="")
            return
        se1 = col['mentions'].sum()
        se2 = col['count'].sum()
        se3 = col['total centrality'].sum()
        se4 = col['average centrality'].sum()
        result = se1 + (se3/se2) + se4
        est_time = 150
        est_result = 70
        if(35<result<est_result):
            length = est_time
        elif(30<result<est_result/2):
            length = est_time-20
        elif(70<result<est_result*2):
            length = est_time+20
        elif(140<result<est_result*4):
            length = est_time+10
        else:
            length = est_time - 10
        print("The predicted length of movie ", m, " on the basis of Centrality and Mentions is about ", np.round((length/60), 2),sep="")

#Results Trends in the movie...
    def trends(self, bol):
        df = {}
        for i in range(10):
            df[i] = self.df2[self.df2['year']==2008+i]['emotion'].value_counts().to_frame()
            df[i].columns = [2008+i]
        df_area = pd.concat([df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7],df[8], df[9]], axis=1)
        if(bol):
            print(df_area)
            df_area.transpose().plot.area()
            plt.xlabel("Year")
            plt.ylabel("Range")
            plt.show()
        else:
            return df_area

    def predict(self):
        df_area = self.trends(False)
        print(df_area)
        z = pd.read_csv('./trend_emotion.csv') #Data Preprocessimg
        X = z.iloc[:, :-1]
        y = z.iloc[:, -1]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        predictions_lin = lm.predict(X_test)



        from sklearn import metrics
        result = list()
        result.append(metrics.mean_squared_error(y_test, predictions_lin))
        result = np.array(result)
        new = list()
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neurtal', 'sad', 'suprise']
        print("\n")
        for i in range(7):
            print("Enter value of",emotions[i],":", end="")
            new.append(int(input()))
        result = lm.predict([new])
        print("\nThe predicted year according the values given is ",result[0])

#through the image of the movie
    def image_movie(self, arr):
        for i in range(len(arr)):
            if arr[i] in self.df5.iloc[:, -1].values:
                m = arr[i]
                self.lead_role(m)
                self.characters(m)
                self.plot(m)
                self.year(m)
                self.songs(m)
                self.average_emotion(m, 0)
                self.length_of_movie(m)
                return
        print("Could not find the movie in the dataset. Try another image.")



# Main Code
ext = 0
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
            print("Query does not contain enough parameters/things.")
