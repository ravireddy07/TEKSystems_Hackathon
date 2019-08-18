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

