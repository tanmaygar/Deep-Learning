# Create MLP with 1 hidden layer from scratch
class MLP:
    def __init__(self, x, y, hidden_size=4, lr=0.1) -> None:
        self.x = x
        self.y = y
        self.input_size = x.shape[1]
        self.hidden_size = hidden_size
        self.output_size = 1
        self.lr = lr
        
        # Weights and Biases
        self.w1 = torch.randn(self.input_size, self.hidden_size)
        # print("w1: ",self.w1.shape)
        self.b1 = torch.randn(1) * torch.randn(self.hidden_size) 
        # print("b1: ",self.b1.shape)
        self.w2 = torch.randn(self.hidden_size, self.output_size)
        # print("w2: ",self.w2.shape)
        self.b2 = torch.randn(1)
        
    
    # Signmoid Activation Function
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    # Signmoid Derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Hinge Loss Function
    def hinge_loss(self, y_pred, y):
        # print(y_pred.shape)
        # print(y.shape)
        # print((y_pred * y).shape)
        return torch.max(torch.zeros_like(y_pred), 1 - y_pred * y)
    
    # Square Loss Function
    def square_loss(self, y_pred, y):
        return (y_pred - y) ** 2
    
    # Hinge Loss Derivative
    def hinge_loss_derivative(self, y_pred, y):
        # print(y_pred.shape)
        # print(y.shape)
        return -y * (y * y_pred < 1)
    
    # Square Loss Derivative
    def square_loss_derivative(self, y_pred, y):
        return 2 * (y_pred - y)
    
    # Binary Cross Entropy Loss Function
    def binary_cross_entropy(self, y_pred, y):
        return - y * torch.log(y_pred) - (1 - y) * torch.log(1 - y_pred)
    
    # Binary Cross Entropy Loss Derivative
    def binary_cross_entropy_derivative(self, y_pred, y):
        return  -(y / y_pred) + ((1 - y) / (1 - y_pred))
    
    # Forward Propagation
    def forward(self, x):
        self.z1 = x @ self.w1 + torch.Tensor.repeat(self.b1, x.shape[0], 1)
        # print("z1: ",self.z1.shape)
        
        self.a1 = self.sigmoid(self.z1)
        # print("a1: ",self.a1.shape)
        
        self.z2 = self.a1 @ self.w2 + self.b2
        # print("z2: ",self.z2)
        
        self.a2 = self.sigmoid(self.z2)
        # print("a2: ",self.a2)
        
        # print("w2: ",self.w2)
        # print("w1: ",self.w1)
        return self.a2
    
    # Backward Propagation
    def backward(self, x, y):
        y = y.reshape(-1, 1)
        # print("x shape: ", x.shape)
        
        self.loss_a2 = self.binary_cross_entropy(self.a2, y)
        # print("Loss_a2 shape: ", self.loss_a2.shape)
        
        # Calculate Gradients
        self.dL_d_a2 = self.binary_cross_entropy_derivative(self.a2, y)
        # print("dL_d_a2 shape: ", self.dL_d_a2.shape)
        
        self.da2_dz2 = self.sigmoid_derivative(self.a2)
        # print("da2_dz2 shape: ", self.da2_dz2.shape)
        
        self.dz2_d_w2 = self.a1.T
        # print("dz2_d_w2 shape: ", self.dz2_d_w2.shape)
        
        self.dz2_d_b2 = torch.ones_like(self.z2)
        # print("dz2_d_b2 shape: ", self.dz2_d_b2.shape)

        self.dL_d_w2 = self.dz2_d_w2 @ (self.dL_d_a2 * self.da2_dz2)
        # print("dL_d_w2 shape: ", self.dL_d_w2.shape)
        
        # self.dL_d_b2 = (self.dz2_d_b2 * (self.dL_d_a2.T @ self.da2_dz2)).reshape(-1)
        self.dL_d_b2 = ((self.dL_d_a2 * self.da2_dz2).T @ self.dz2_d_b2).reshape(-1)
        # print("dL_d_b2 shape: ", self.dL_d_b2.shape)
        
        self.da1_dz1 = self.sigmoid_derivative(self.a1)
        # print("da1_dz1 shape: ", self.da1_dz1.shape)
        
        self.dz1_dw1 = x
        # print("dz1_dw1 shape: ", self.dz1_dw1.shape)
        
        self.dz1_x = self.w1
        
        self.dz2_d_a1 = self.w2
        
        # print("dz2_d_a1 shape: ", self.dz2_d_a1.shape)
        # print("dz1_x shape: ", self.dz1_x.shape)
        # self.dz1_d_b1 = torch.ones_like(self.a1)
        # print("dz1_d_b1 shape: ", self.dz1_d_b1.shape)
        
        # print("ter: ", ((self.dL_d_a2 * self.da2_dz2) @ self.w2.T).shape)
        self.dL_dw1 =  self.dz1_dw1.T @ (((self.dL_d_a2 * self.da2_dz2) @ self.w2.T) * self.da1_dz1)
        
        # print("dL_dw1 shape: ", self.dL_dw1.shape)
        # print("ter ",(((self.dL_d_a2 * self.da2_dz2) @ self.w2.T) @ self.da1_dz1.T).shape)
        # print((((self.dL_d_a2 * self.da2_dz2).T @ (self.da1_dz1 * self.dz1_d_b1))).T.shape)
        # print("ter ",(((self.dL_d_a2 * self.da2_dz2) @ self.w2.T) * self.da1_dz1).shape)
        self.dL_db1 = (((self.dL_d_a2 * self.da2_dz2) @ self.w2.T) * self.da1_dz1).sum(axis=0)
        
        # print("dL_db1 shape: ", self.dL_db1.shape)
        
        # Updating Weights and Biases
        self.w2 -= self.lr * self.dL_d_w2
        self.b2 -= self.lr * self.dL_d_b2
        self.w1 -= self.lr * self.dL_dw1
        self.b1 -= self.lr * self.dL_db1
        
        
    # Training the Model
    def train(self, epochs=100):
        loss = []
        for epoch in range(epochs):
            self.forward(self.x)
            # print("W1 in forward: ", self.w1)
            # print("W2 in forward: ", self.w2)
            # print("B2 in forward: ", self.b2)
            # print("A2 in forward: ", self.a2)
            self.backward(self.x, self.y)
            # print("W1 in backward: ", self.w1)
            # print("W2 in backward: ", self.w2)
            # print("B2 in backward: ", self.b2)

            loss.append(self.loss_a2.mean())
            if epoch % 10 == 0:
                print("Epoch: ", epoch, " Loss: ", self.loss_a2.mean())
        # print("Final Predicted: ", self.forward(self.x))
        # print("Ground Truth: ", self.y)
        # print("z2: ", self.z2)
        
        # Plotting the Loss Curve
        plt.figure(figsize=(20, 5))
        plt.xticks(np.arange(0, epochs + 10, 10))
        plt.plot(loss)
        plt.xlim(-1, epochs)
        plt.ylim(min(loss), max(loss) + 0.1)
        plt.yticks(np.arange(0, max(loss) + 0.1, 0.1))
        plt.title("Loss vs Epochs")
        plt.show()
        return    
        
    def predict(self, x, threshold=0.5):
        predicted = self.forward(x)
        # print(torch.round(predicted))
        # apply threshold of 0.6 for label
        # print(predicted)
        
        return predicted > threshold

    
    def accuracy(self, x, y):
        # print(self.predict(x).shape)
        y = y.reshape(-1, 1)
        return torch.sum(self.predict(x) == y) / y.shape[0]
