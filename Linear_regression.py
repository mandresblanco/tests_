import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('longley.csv')
#df.columns

#plt.scatter(df.GNP, df.Employed)
k=np.corrcoef(df.Employed,df.GNP)[0,1]
print(k)
y=df.Employed
X=df.GNP
X=sm.add_constant(X)


lr_model = sm.OLS(y,X).fit()
print(lr_model.summary())
#Employed = coeff*GNP + const
# Build the regression model using OLS (ordinary least squares)
lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())
# We pick 100 points equally spaced from the min to the max
X_prime = np.linspace(X.GNP.min(), X.GNP.max(), 100)
X_prime = sm.add_constant(X_prime) # Add a constant as we did before

# Now we calculate the predicted values
y_hat = lr_model.predict(X_prime)
plt.scatter(X.GNP, y) # Plot the raw data
plt.xlabel('Gross National Product')
plt.ylabel('Total Employment')
# Add the regression line, colored in red
plt.plot(X_prime[:, 1], y_hat, 'red', alpha=0.9)
#If you see something strange in your plots from the above code, chances are your plotting environments
#are getting messed up. To address that, change the code for your plotting as shown below:
# If you see something strange in your plots from the above code, chances are your plotting environments
# are getting messed up. To address that, change the code for your plotting as shown below:
# plt.figure(1)
# plt.subplot(211)
# plt.scatter(df.Employed, df.GNP)
# plt.subplot(212)
# plt.scatter(X.GNP, y) # Plot the raw data
# plt.xlabel(“Gross National Product”)
# plt.ylabel(“Total Employment”)
# # Add the regression line, colored in red
# plt.plot(X_prime[:, 1], y_hat, ‘red’, alpha=0.9)
# Essentially, we are creating separate spaces to display the original scatterplot, and the new scatterplot with
# the regression line