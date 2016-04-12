from ffnn import FFNN
import numpy as np
from sklearn.datasets import make_moons

nn = FFNN(2,[5],2,noise_scale=.1)

I = np.identity(2)
X_data,y = make_moons(200,.17)
y_data = np.array([I[value] for value in y]) 
fit_score = nn.fit(X_data,y_data)
for fit_line in fit_score:
    print(fit_line)
print(nn.accuracy(X_data,y_data))


