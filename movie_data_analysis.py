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
       
        # creating word cloud
        if(n==0):
            self.create_wordcloud(col)
        
        if(n==0):
            # genre of the film
            self.genre(m)
        

def genre(self, m):
        col = self.df2[self.df2['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the genre.", sep="")
            return
        se = col['emotion'].value_counts()
        gen = se[se == max(se.values)].index[0]

