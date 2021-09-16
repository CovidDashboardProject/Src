
from src.Data.data import Covid_Data
from src.LSTM_model.model import Covid_pred_model
from src._helper.helper import plot_data,get_pred
import tensorflow as tf
import numpy as np


class Core:

    def __init__(self,
                    n_steps = 30, 
                    optimizer='adam',
                    loss='mape',
                    batch_size = 128,
                    epochs = 100,
                    steps_per_epoch = 10,
                    test_split = 0.1,
                    validation_steps = 5,
                    model_path = ''):

        self.n_steps = n_steps

        self.optimizer = optimizer
        self.loss = loss

        self.batch_size = batch_size

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.test_split = test_split
        self.validation_steps = validation_steps

        if model_path == '' : model_path = 'Data/saved_model/'

        early_stopings = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='min')
        checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='loss', save_best_only=True, mode='min', verbose=0,save_format = 'tf')
        self.callbacks=[early_stopings,checkpoint] 


    def get_data(self):

        self.C_data = Covid_Data()

        self.atomic_data = self.C_data.get_realtime_data()

        self.X,self.y = self.C_data.preprocess_data(self.atomic_data,self.n_steps)

        data = self.C_data.train_test_split(self.X,self.y,self.test_split)

        return data
    
    def get_model(self):

        model = Covid_pred_model((self.n_steps,1))

        model.compile(optimizer=self.optimizer, loss=self.loss)

        _ = model(tf.zeros([1,self.n_steps,1]))

        return model

    def train(self,data,model,verbose = 1):

        print('Starting the tarnning ... ')
        history = model.fit(data['train'][0],
                    data['train'][1],
                    epochs=self.epochs,
                    steps_per_epoch=self.steps_per_epoch,
                    validation_data=data['test'],
                    validation_steps=self.validation_steps,
                    verbose=verbose,
                    callbacks=self.callbacks)

        print('\n\n')
        plot_data([history.history['loss'],history.history['val_loss']],
                    title = 'error_graph',
                    ylabel = 'loss',
                    xlabel = 'epoch',
                    legend = ['train loss', 'validation loss'])

        return history

    def show_results(self,model,data,future_date = 10):

        Orignal_data = self.atomic_data
        Prediction_data = [None for i in range(self.n_steps)] + [self.C_data.scaler.inverse_transform(np.array([i[0]]).reshape(1,-1))[0][0] for i in model(self.X)]

        plot_data([Orignal_data,Prediction_data],
                title = 'actual vr pred',
                ylabel = 'cases',
                xlabel = 'time',
                legend = ['actual', 'pred'])


        Prediction_data = get_pred(self.C_data,model,Orignal_data,future_date,self.n_steps)

        print("\n\n\n")

        plot_data([Orignal_data,Prediction_data],
                title = 'future_pred',
                ylabel = 'cases',
                xlabel = 'time',
                legend = ['actual', 'pred'])



        






