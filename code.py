from sklearn import linear_model
import numpy as np

#australia:
x = np.array([[1.08,1.00344,1,0,0.6],[1.08,1.00344,1,0,0.6],[1.08,1.00344,0,0,0.6],[1.08,1.00344,1,0,0.6],[1.08,1.00344,0,0,0.6]])
Y = np.array([0,0,0,0,1])
model = linear_model.LogisticRegression()
model.fit(x,Y)
print ('australia:')
a = [1.08,1.00344, 0, 0, 0.6]
print ('prediction : ',model.predict(a))
b = [1.08, 1.00344, 1, 0, 0.6]
print ('prediction : ',model.predict(b))
print ('Co-efficient',model.coef_)

#england:
print ('england:')
x = np.array([[1.122,1.069,1,1,0.6],[1.122,1.069,0,1,0.6],[1.122,1.069,0,1,0.6],[1.122,1.069,1,1,0.6],[1.122,1.069,1,1,0.6]])
Y = np.array([1,1,0,0,1])
model = linear_model.LogisticRegression()
model.fit(x,Y)
a = [1.122 ,1.069, 0, 0 ,0.6]
print ('prediction : ',model.predict(a))
b = [1.122 ,1.069 ,1, 0, 0.6]
print ('prediction : ',model.predict(b))
print ('Co-efficient',model.coef_)

#sa:
print ('south africa:')
x = np.array([[0.8966,0.99,1,1,0.6],[0.8966,0.99,0,1,0.6],[0.8966,0.99,1,1,0.6],[0.8966,0.99,0,1,0.6],[0.8966,0.99,1,1,0.6]])
Y = np.array([0,1,0,1,0])
model = linear_model.LogisticRegression()
model.fit(x,Y)
a = [0.8966, 0.99, 0, 0, 0.6]
print ('prediction : ',model.predict(a))
b = [0.8966, 0.99 ,1 ,0 ,0.6]
print ('prediction : ',model.predict(b))
print ('Co-efficient',model.coef_)


