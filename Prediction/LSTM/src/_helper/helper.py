import matplotlib.pyplot as plt
import numpy as np

def plot_data(Data,title,ylabel,xlabel,legend):

    plt.figure(figsize=(16,9))

    for data in Data:
      plt.plot(data)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)

    plt.show() 


def get_pred(C_data,model,data,num_days,n_steps):
  
  pred = []
  if type(data) == np.ndarray:
    data = list(data)

  prv_pred_len = len(data)

  for _ in range(num_days):

    x = np.array(data[-n_steps:]).reshape(-1,1)

    x = C_data.scaler.transform(x)

    y_ = C_data.scaler.inverse_transform(np.array(model(x.reshape(1,n_steps,1))).reshape(1,-1))[0][0]

    data.append(y_)

    pred.append(y_)
  
  return [None for i in range(prv_pred_len)] + pred