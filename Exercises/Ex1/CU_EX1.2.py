# pip install "../Requirements.txt"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d ## check with avishai
from sklearn.linear_model import LinearRegression


# print(f"Matplotlib version: {plt.__version__}")
print(f"NumPy version: {np.__version__}")




####################################################################################################
print("Getting Started with Numpy")
a = np.array([1,2,3,4,5])
b = np.array([1,2,3])
c = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a, b, c)
print(a.shape, b.shape, c.shape) #.shapes shows the shape of the array (rows, columns)
print("\n\n")


####################################################################################################
print("Reshaping")
a = a.reshape(1,-1) #1 row and 5 columns
b = b.reshape(1,-1)
print(a, b)
print(a.shape, b.shape)
print("\n\n")


####################################################################################################
print("Indexing")
c = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,-8,9]])
print(c.shape)
print(c)
print("\n\n")



####################################################################################################
# Transpose
print("Transpose")
print(c.T)
print("---")

# dot product
print(c.dot(c.T)) # c * C.T
print("---")
print(c.T.dot(c)) # = np.matmul(c.T, c)
print("---")

# multiply
print("Multiplication")
print(np.multiply(c, c)) # element wise multiplication
print("\n\n")


####################################################################################################
# concatenation
print("Concatenation")
a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])

a = a.reshape(-1,1)
b = b.reshape(-1,1)
print(a.shape, b.shape)
c = np.concatenate((a,b), axis=1) # axis=0 - vertical | axis=1 - horizontal
print(c)
print(c.shape)
print("\n\n")




####################################################################################################
b = np.copy(a)  #Creates a copy of the memory       
print(a, b)
b[2] = -1000
print("---")
print(a)
print(b)


d = a       #Saves a pointer to data of a . if a changes the b is changed too
d[2] = 1000
print(a)
print(d)
print("\n\n")




####################################################################################################
a1 = np.zeros((10,4)) #Creates a 10x4 matrix with zeros 4 columns and 10 rows
a2 = np.ones((10,4))
print(a1)
print("---")
print(a2)
print("\n\n")


####################################################################################################
print("Slicing")
a = np.array([1,2,3,4,5])
v = a[:2] #list all until the 2nd element
print(v)
print("---")
v = a[1:3] #list from 1 to 3
print(v)
print("---")
v = a[:-3] #list all until the 3rd element 3rd element from the end
print(v)
print("---")
c = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,-8,9]])
v = c[:2,:-1] #return 2 rows with columns without the last element
print(v)
print("\n\n")




####################################################################################################
print("Linear Algebra")
a = np.array([1,3,4])
b = np.array([9,3,4])
a = np.linalg.norm(a - b) #numpy.liniearAlgebra.normall ||a-b|| = sqrt((a-b)^2)
print(a)
print(np.linalg.det(c[1:,:]))
print(np.linalg.det(c[1:,:]))
print("\n\n")




####################################################################################################
print("Statistics")
a = np.array([1,2,3,4,5])

# statistics
print(np.sum(a), a.sum())
print(np.mean(a), a.mean())
print(np.std(c, axis=1))
print("---")

# inserting and delete
a = np.insert(a, 3, -1) #insert -1 at the 3rd index
print(a)
a = np.delete(a, 3) #delete the 3rd index
print(a)
print("---")

c = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,-8,9]])
b = np.array([9,3,4]).reshape(1,-1)
c = np.insert(c, 2, b, axis=0)
print(c)
print("\n\n")




####################################################################################################
print("Plotting")
x = np.linspace(0, 100, 1000) #creates a list of 1000 elements from 0 to 100
y1 = x**2 / 100
y2 = -x + 3

f = plt.figure(figsize=(10,5))
plt.plot(x, y1, '-k', label = 'x**2 / 100', linewidth = 2)
plt.plot(x, y2, ':b', label = '-x + 3', linewidth = 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Functions')
plt.legend()

plt.show()



print("Subplots")
fig, ax = plt.subplots(1, 2, figsize=(6,5))#, sharey=True) #ad is an array for each sub plot
ax[0].plot(x, y1, '-k', linewidth = 2)
ax[0].set_title('Function 1')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].plot(x, y2, '-k', linewidth = 2)
ax[1].set_title('Function 2')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

plt.show()





print("Scatter Plot")
xs = np.array([1,1])
xg = np.array([8,10])
S = np.linspace(0, 1, 20)   # line = xs + s * (xg-xs)
X = np.array([xs + s * (xg-xs) for s in S])

circ = [8,4]
ro = 2
pts = np.array([[0,2], [4,5], [1,4.5]])

fig, ax = plt.subplots()
ax.plot(X[:,0], X[:,1], '-s')
circle = plt.Circle((circ[0], circ[1]), ro, color='g') #Circle((x,y), radius, color)
rec = plt.Rectangle((2,7), 3, 2, fc='gray',ec="red") #Rectangle((x,y), width, height, facecolor, edgecolor)
p = plt.Polygon(pts, fc = 'magenta') #Polygon([points], facecolor)

ax.add_artist(circle) 
ax.add_artist(rec)
ax.add_artist(p)

plt.axis('equal') #makes sure attributes are equal (Axis are scaled equally)




####################################################################################################
r = 2.0
P = []
for _ in range(100000): #loop with no purpose to use the index
    
    # a -> b
    # np.random.random() * (a-b) + b
    theta = np.random.random() * (2 * np.pi) - np.pi
    phi = np.random.random() * 2 * np.pi + 0

    p = np.array([r * np.sin(theta) * np.cos(phi), 
                  r * np.sin(theta) * np.sin(phi), 
                  r * np.cos(theta)])
    P.append(p)
P = np.array(P)
print(P.shape)

Z = np.linspace(-2, 2, 1000)    #red spiral
X = np.sin(14*Z)                #red spiral
Y = np.cos(14*Z)                #red spiral

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(P[:,0], P[:,1], P[:,2], '.k', markersize = 0.5)
ax.plot3D(X, Y, Z, 'red')

plt.show()




####################################################################################################
print("Linear Regression")
"""
The general flow of fitting and using a model is:

Generating / Load training data
Fit model
Predict / Test new data samples
Evaluate and visualize if possible
"""

# Generate training data
rng = np.random.RandomState(42)

print(rng)
x = np.linspace(0, 10, num=2000).reshape(-1,1)
print(x)

y = 3.2 * x + rng.normal(scale=x / 2) #Ax + b    : b random values. y size 2000x1
print(y)


model = LinearRegression() #Linear Regression model 
model.fit(x, y)


y_predicted = model.predict(x)
y_predicted


plt.figure()
plt.plot(x, y, 'o', alpha=0.5, markersize=1, label = 'Data')
plt.plot(x, y_predicted, '-k', label = 'Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()