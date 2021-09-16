import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Covid_Data:

  def __init__(self,data_url = None ):

    if not data_url:
      self.data_url = 'http://www.cessi.in/coronavirus/images/model_output/pmc_1.csv'
    else:
      self.data_url = data_url

    self.scaler = MinMaxScaler()


  def get_realtime_data(self,path = 'Data/'):

    data = pd.read_csv(self.data_url).drop(columns = ['Unnamed: 0'])

    data.to_csv(path+'Pune_data.csv')

    daily_confirmed = np.array(data['dailyconfirmed'].tolist())

    return daily_confirmed
    

  def get_sample(self,data, n_steps):

    X, Y = [], []

    for i in range(len(data)):

      sam = i + n_steps
      
      if sam > len(data)-1:
        break

      x, y = data[i:sam], data[sam]

      X.append(x)

      Y.append(y)

    return np.array(X), np.array(Y)

  
  def preprocess_data(self,data,n_steps = 30):

    data = data.reshape(-1, 1)

    self.scaler.fit(data)

    data = self.scaler.transform(data)

    X, y = self.get_sample(data, n_steps)

    return X,y

  def train_test_split(self,X,y,test_size):


    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=42)

    data = {'train' : (X_train,y_train),
            'test' : (X_test,y_test)}

    return data
      


