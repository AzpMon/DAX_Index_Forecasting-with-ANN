from torch.utils.data import Dataset
import pandas as pd 
import torch
from sklearn.preprocessing import MinMaxScaler 
from torch.utils.data import TensorDataset, DataLoader


class TimeSeries_Dataset(Dataset):
    def __init__(self, yahoo_df, fecha_inicial, fecha_final, len_window, train_proportion = 0.7):
        self.X, self.y = self.lstm_tensor_data(yahoo_df, fecha_inicial, fecha_final, len_window)[0]
        self.LSTM_df = self.lstm_tensor_data(yahoo_df, fecha_inicial, fecha_final, len_window)[1]
        self.len_window = len_window 
        self.fecha_inicial = fecha_inicial
        self.fecha_final = fecha_final 
        self.yahoo_df = yahoo_df
        self.samples = self.y.shape[0]
        self.dates = self.LSTM_df.index
        self.train_proportion = train_proportion
        
        #Train-Test split
        self.train_proportion = train_proportion
        train_length =int( self.samples * self.train_proportion )
        
        #Train Data
        self.X_train = self.X[0:train_length]
        self.y_train = self.y[0:train_length]
        self.train_dataset = TensorDataset(self.X_train, self.y_train )
        self.train_loader = DataLoader(self.train_dataset, shuffle=False)
        self.train_dates =self.LSTM_df.index[:int(self.samples * self.train_proportion)]

        #Test Data
        self.X_test = self.X[train_length:]
        self.y_test = self.y[train_length:]
        self.test_dataset  =  TensorDataset(self.X_test, self.y_test )
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)
        self.test_dates =self.LSTM_df.index[int(self.samples * self.train_proportion):]


        

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Se regresa un registro de datos junto con su etiqueta

    def __len__(self):
        return self.samples

    def lstm_tensor_data(self, yahoo_df, fecha_inicial, fecha_final, len_window):
        def create_lstm_data_no_dataTime(yahoo_df, fecha_inicial, fecha_final, len_window):
            df = yahoo_df.copy()
            if fecha_inicial not in df.Date.to_list():
                raise Exception(f'{fecha_inicial} no está registrada en el DataFrame')
            
            if fecha_final not in df.Date.to_list():
                raise Exception(f'{fecha_final} no está registrada en el DataFrame')
            
            if df.Date.to_list().index(fecha_inicial) < len_window:
                raise Exception(f'No hay datos suficientes, dado la len_window={len_window} y fecha inicial {fecha_inicial}')
                

            init_index = df[df["Date"] == fecha_inicial].index.tolist()[0]
            final_index = df[df["Date"] == fecha_final].index.tolist()[0]
            

            torch_data = torch.zeros(size=(final_index - init_index +1, len_window+1), requires_grad=False)
                
            
            for i in range(final_index-init_index+1):
                torch_data[i,len_window] = df["Close"].loc[init_index+i]
                for window_day in range(len_window):
                    torch_data[i, window_day] = df["Close"].loc[(init_index+i) - (len_window-window_day) ]
            
            #Se crea el DataFrame asociado a los datos de entrenamiento de la LSTM
            df_columns = [f'Close_t_minus_{n-1}' for n in range(len_window+1,1,-1)] + ["Close_target"]
            lstm_df = pd.DataFrame(data = torch_data, columns = df_columns)
            lstm_df["CloseTargetDate"] = [df["Date"].iloc[init_index + n] for n in range(final_index-init_index+1) ]
            
            lstm_df["CloseTargetDate"]=pd.to_datetime(lstm_df["CloseTargetDate"])
            lstm_df.index = lstm_df.pop('CloseTargetDate')
        
            lstm_tensor = torch.tensor(lstm_df.values, requires_grad=False)
            return lstm_df, lstm_tensor
        

        lstm_df, lstm_tensor = create_lstm_data_no_dataTime(yahoo_df,fecha_inicial, fecha_final, len_window)

        #Se re-escalan los datos 
        scaler = MinMaxScaler(feature_range=(-1,1))
        transform_data = scaler.fit_transform(lstm_tensor)
        X =  transform_data[:, :len_window].reshape((-1, len_window, 1))
        y =  transform_data[:, len_window].reshape((-1, 1))

         

        return [(torch.tensor(X, dtype=torch.float32),torch.tensor(y, dtype=torch.float32)), lstm_df]  