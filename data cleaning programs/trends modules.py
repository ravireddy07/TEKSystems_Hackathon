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
        
