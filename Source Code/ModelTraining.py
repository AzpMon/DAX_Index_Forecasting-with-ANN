import torch 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

class ModelTraining(object):
    def __init__(self, model, loss_fn, optimizer):
        
        #Atributos de la clase
        self.model = model 
        self.loss_fn = loss_fn
        self.optimizer = optimizer 
        self.device = 'cuda' if torch.cuda.is_available() else  'cpu'
        
        #Se envía el modelo al device especificado (cuda o cpu)
        self.model.to(self.device)
        

        #Atributos que se computarán internamente 
        self.losses=[]
        self.test_losses=[]
        self.total_epochs=[]
        
        #Se crea la función de entrenamiento 
        self.train_step_fn = self._make_train_step_fn()
        
        #Se crea la función de prueba del modelo y su fn. pérdida
        self.test_step_fn = self._make_test_step_fn()
        
    
    #El usuario puede definir el train_loader y test_loader    
    def set_loaders(self, train_loader, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
       
    ############################################################################
    def _make_train_step_fn(self):
        #Se utilizan los atributos de la misma clase
        
        #Se define la función que realiza un paso en la optimización del modelo
        def perfom_train_step_fn(x,y):
            
            #Se pone al modelo en modo entrenamiento
            self.model.train()
            
            #PASO 1. Calcular la predicción del modelo (Forward pass)
            y_hat = self.model(x)
            
            #PASO 2: Calcular la función de pérdida 
            loss = self.loss_fn(y_hat, y)
            
            #PASO 3: Calcular los gradientes de los pesos y los sesgos
            loss.backward()
            
            #PASO 4: Actualizar los parámetros utilizando los gradientes y el learning_rate
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            #Se returna el valor de la función de pérdida
            return loss.item() #Se utiliza .item() porque es un tensor de 1 dimensión
        
        return perfom_train_step_fn
    
    ############################################################################
    def _make_test_step_fn(self):
        def perform_test_step_fn(x,y):
            #Se pone el modelo en modo "evaluar"
            self.model.eval()
            
            #PASO 1. Calcular la predicción del modelo (Forward pass)
            y_hat = self.model(x)
            
            #PASO 2: Calcular la pérdida
            loss = self.loss_fn(y_hat,y)
            
            return loss.item()
        return perform_test_step_fn
    
    
    ############################################################################
    def _mini_batch(self, test=False):
        #Se configuran los DataLoaders (de pytorch) y las funciones de paso (step_fn)
        if test ==True:
            data_loader = self.test_loader
            step_fn = self.test_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        
        if data_loader is None:
            return None
        
        
        #MiniBatch Loop
        mini_batch_losses=[]
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        
        loss = np.mean(mini_batch_losses)
        
        return loss
    
    
    
    
    ############################################################################
    #Para asegurar la reproducibilidad (¿?)
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    ############################################################################
    def train(self, n_epochs, seed=42):
        # Para asegurar la reproducibilidad del proceso de entrenamiento (¿?)
        self.set_seed(seed)
        
        
        # Bucle de entrenamiento
        for epoch in range(n_epochs):
            # Loop interno: Se realiza el entrenamiento usando mini_batches
            loss = self._mini_batch(test=False)
            self.losses.append(loss)

            # Prueba
            # no se requiere calcular gradientes
            with torch.no_grad():
                # Se evalua el modelo utilizando mini-batches
                test_loss = self._mini_batch(test=True)
                self.test_losses.append(test_loss)

            #Actualizar el número de épocas después de cada iteración
            self.total_epochs.append(epoch + 1)
            
    def predict(self, x):
        
        # Set it to evaluation mode for predictions
        self.model.eval()
        # Take a Numpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and use model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detach it, bring it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()
    

        
    
    def plot_MSE(self, Title=''):
        
        fig = plt.figure(figsize=(12, 4))
        plt.plot(self.losses, label='Training MSE', c='darkred')
        if self.test_loader:
            plt.plot(self.test_losses, label='Test MSE', c='midnightblue', ls = '--')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title(Title)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
        
        return fig
        
        
    def plot_forecasting(self, X,Y, dates, Title='', train_or_test = 'train'):
        fig = plt.figure(figsize=(12, 4))
        predictions = self.predict(X)
 
        if train_or_test == 'train':
            plt.plot(dates, predictions, label = "Predictions", c = 'red', lw = 0.9)
            plt.plot(dates,Y.numpy(), label = "Real values",c = 'blue', lw = 0.9)
            
        else:
            plt.plot(dates, predictions, label = "Predictions", c = 'tomato', lw = 0.9)
            plt.plot(dates,Y.numpy(), label = "Real values",c = 'darkcyan', lw = 0.9)
            
            
        plt.xlabel('Date')
        plt.ylabel("Scaled Close")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.title(f'{Title}\n(Forecasting from {dates[0]} to {dates[-1]})')
        plt.show()
        
        return fig