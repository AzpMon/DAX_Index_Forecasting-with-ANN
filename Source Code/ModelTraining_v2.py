import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelTraining(object):
    def __init__(self, model, loss_fn, optimizer):
        # Atributos de la clase
        self.model = model 
        self.loss_fn = loss_fn
        self.optimizer = optimizer 
        self.device = 'cuda' if torch.cuda.is_available() else  'cpu'
        
        # Se envía el modelo al device especificado (cuda o cpu)
        self.model.to(self.device)
        
        # Atributos que se computarán internamente 
        self.losses=[]
        self.test_losses=[]
        self.total_epochs=[]
        
        # Se crea la función de entrenamiento 
        self.train_step_fn = self._make_train_step_fn()
        
        # Se crea la función de prueba del modelo y su fn. pérdida
        self.test_step_fn = self._make_test_step_fn()
        
    # El usuario puede definir el train_loader y test_loader    
    def set_loaders(self, train_loader, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    ############################################################################
    def _make_train_step_fn(self):
        def perfom_train_step_fn(x,y):
            self.model.train()
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perfom_train_step_fn
    
    ############################################################################
    def _make_test_step_fn(self):
        def perform_test_step_fn(x,y):
            self.model.eval()
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat,y)
            return loss.item()
        return perform_test_step_fn
    
    ############################################################################
    def _mini_batch(self, test=False):
        if test ==True:
            data_loader = self.test_loader
            step_fn = self.test_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        
        if data_loader is None:
            return None
        
        mini_batch_losses=[]
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        
        loss = np.mean(mini_batch_losses)
        return loss

    ############################################################################
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    ############################################################################
    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        for epoch in range(n_epochs):
            loss = self._mini_batch(test=False)
            self.losses.append(loss)
            with torch.no_grad():
                test_loss = self._mini_batch(test=True)
                self.test_losses.append(test_loss)
            self.total_epochs.append(epoch + 1)
            
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()
        
    ############################################################################
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
        
    ############################################################################
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
    

    
    ############################################################################
    def compute_metrics(self, X_train, y_train, X_test, y_test):
        

        
        # Predictions
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        
        # Metrics calculations
        metrics = {}
        
        def calculate_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)
            return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'RMSPE': rmspe, 'MAPE': mape, 'R²': r2}
        
        metrics['train'] = calculate_metrics(y_train, y_train_pred)
        metrics['test'] = calculate_metrics(y_test, y_test_pred)
        
        return metrics
