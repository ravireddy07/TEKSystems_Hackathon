 def predict(self):
        df_area = self.trends(False)
        print(df_area)
        # Data-Preprocessing
        z = pd.read_csv('./trend_emotion.csv')
        X = z.iloc[:, :-1]
        y = z.iloc[:, -1]
        # Spliting Data 
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Linear Regression
        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        predictions_lin = lm.predict(X_test)
       
