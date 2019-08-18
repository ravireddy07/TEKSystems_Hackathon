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


def genre(self, m):
        col = self.df2[self.df2['movie']==m]
        if(col.empty):
            print("The movie ", m, " is not found in the database. Cannot find the genre.", sep="")
            return
        se = col['emotion'].value_counts()
        gen = se[se == max(se.values)].index[0]

