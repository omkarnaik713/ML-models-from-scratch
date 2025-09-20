import pandas as pd 

data = pd.read_csv()

def loss_function(m,b, points) :
    total_error = 0 
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        
        total_error += (y-(m*x + b)) ** 2 
        
    return total_error/len(points)

def gradient_descent(m,b, points, L):
    n = len(points)
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        
        m_grad = (-2/n) * x *  (y - (m*x + b))
        b_grad = (-2/n) * (y- (m*x + b))
    
    m = m - (L* m_grad)
    b = b - (L*b_grad)
    return m , b 


m = 1 
b = 1
L = 0.0001
epochs = 300

for i in range(epochs):
    m,b = gradient_descent(m,b,data,L)
    
print(m,b)
