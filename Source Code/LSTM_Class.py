import torch 
import torch.nn as nn 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = 1  #El número de caracteristicas de los datos de entrenamiento (Close value)
        self.hidden_size = hidden_size #El número de outputs que se quieren en el modelo


        #REDES NEURONALES A UTILIZAR
        #RNN-LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True)
        
        #Obs: A pesar de utilizar 
        self.fully_connected = nn.Linear(hidden_size, 1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        
        batch_size = x.size(dim=0)

        #Se crea el vector h(t=0), vector histórico inicial (hidden state)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        #Se crea el vector c(t=0), vector del estado de la celda (para facilitar la retropropagagación del gradiente)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        
        #Ingrea el input x  a travez de la red neuronal LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        #Al final, se utiliza  otra red neuronal (fully connected) para la salida (out) de la LSTM
        prediction = self.fully_connected(out[:, -1, :])
        return prediction
 