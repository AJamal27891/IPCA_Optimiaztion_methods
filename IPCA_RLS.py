import numpy as np

class IPCA_RLS:
    def __init__(self, n_components, forgetting_factor=0.99, regularization=1e-6):
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization
        self.W = None
        self.P = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
            
        # Initialize weight matrix and inverse correlation matrix
        self.W = np.random.randn(n_features, self.n_components)
        self.P = np.eye(self.n_components) / self.regularization
            
        # Iterate over input vectors and update weight matrix and inverse correlation matrix
        for i in range(n_samples):
            x = X[i]
            y = np.dot(self.W.T, x).reshape(-1, 1)  # ensure y is a 2D array
            e = x - np.dot(self.W, y)
                
            # Update inverse correlation matrix using Sherman-Morrison formula
            PyyP = np.dot(np.dot(self.P, y), y.T).dot(self.P)
            alpha = 1.0 / (self.forgetting_factor + np.dot(y.T, np.dot(self.P, y)))
            self.P -= alpha * PyyP
                
            # Update weight matrix using RLS update rule
            delta_W = np.outer(e, y.T).dot(self.P)
            self.W += delta_W * self.regularization
        
    def transform(self, X):
        return np.dot(X, self.W)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)