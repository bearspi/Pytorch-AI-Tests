import torch
from torch import nn
import matplotlib.pyplot as plt

mps_device = torch.device('mps:0' if torch.has_mps else 'cpu')


# Eğim
weigth = 0.69

#Başlangıç Noktası
bias = 0.31 

#Verilerin Başlangıç Sayısı
start = 0

#Verilerin Bitiş Sayısı
end = 1

#Veriler Arasındakı Sayı
step = 0.02

#Kaç kere modelin eğitileceği
epochs = 200


#Verileri oluşturuyoruz
X = torch.arange(start=start, end=end, step=step).unsqueeze(1)

#Lineer Regrasyon formülümüz ile X'i y'ye çeviriyoruz
y = weigth * X + bias

#Verilerimizin yüzde kaçını eğitim için kullanıcağımızı seçiyoruz
train_split = int(0.8 * len(X))

#Verilerimizi yukardaki yüzdeliğe göre eğitim ve test olarak bölüyoruz
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#Bura önemli değil isteyen baksın çizim için :)
def plot_predictions(train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="y", s=4, label="Testing data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14});
    plt.show()

#Modelimizi tanımlıyoruz ve anası nn.Module olacak şekilde ayarlıyoruz ki tanımlanan fonksiyonları kullanabilelim
class LinearRegressionModel(nn.Module):
    
    #Başlangıç Fonksiyonu
    def __init__(self):
        super().__init__()
        
        #İlk parametremiz olan eğimi rastgele bir şekilde tanımlıyoruz
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
        #Aynısını ikinci paremetremiz olan başlangıç noktasınada uyguluyoruz
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    #Her kompitasyonda yapılacak işlemi belirtiyoruz
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

#Rastgelelik olmasın diye tohumu ayarlıyoruz (opsiyonel)
torch.manual_seed(81)

#Modeli oluşturuyoruz
model_0 = LinearRegressionModel()

#Bir kaybetme foksiyonu ekliyoruz bunu hata yapınca hatanın ne kadar büyük olduğunu hesaplayan bir fonksiyon gibi hayal edebilirsiniz ben burda MHO seçtim yani mutlak hatanın ortalaması.
loss_fn = nn.L1Loss()

#Bir optimzör seçiyoruz ben burda SGD algoritmasını seçtim buda sürekli yokuş aşığı giderek train_y ile tahmin ettiği verinin farkını azaltmaya çalışıyor lr her paremetreyi ise kaçar kaçar azaltacağını belirtiyor 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

#Bir öğrenme döngüsü açıyoruz
for epoch in range(epochs):
    
    #Modeli öğretme modu alıyoruz
    model_0.train()
    
    #X_traini kullanarak y'yi tahmin etmeye çalışıyoruz.
    y_pred = model_0(X_train)
    
    #Kaybetme Fonksiyonu ile hatayı hesaplıyoruz
    loss = loss_fn(y_pred, y_train)
    
    #Gradyantları sıfırlıyor
    optimizer.zero_grad()
    
    
    loss.backward()
    
    #Optimayzör kullanarak bir adım atlıyor ve parametreleri değiştiriyor.
    optimizer.step()
    
    #her 10 öğrenme döngüsünde tetikleniyor ve test verileriyle test yapıp onların hata paylarını öğrenip konsola yazıyor (opsiyonal)
    if epoch %  10 == 0:
        model_0.eval()

        with torch.inference_mode():
            y_predictions = model_0(X_test)
            test_loss = loss_fn(y_predictions, y_test)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")







