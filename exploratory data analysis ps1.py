import pandas as pd 
import numpy as np 
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

dataset = pd.read_csv("G:\College Work\SEMESTER VI\Cyber Security Analytics Laboratory\Datasets\Obfuscated-MalMem2022.csv")

#General Info about the dataset
#print(dataset.head)

#Different Attributes
#print(dataset.columns)

N = 15 # No of records taken to work upon
#Selecting 6 columns for further work
selected_cols = ['Class','callbacks.ncallbacks','dlllist.avg_dlls_per_proc','pslist.avg_threads','handles.nevent','malfind.protection','svcscan.nservices']
#selected_cols = ['pslist.nppid','dlllist.avg_dlls_per_proc','ldrmodules.not_in_mem','psxview.not_in_deskthrd','svcscan.kernel_drivers','callbacks.ncallbacks']

dataset_random_15 = dataset.sample(n=N)[selected_cols]
print(dataset_random_15)
print("----------------------------------------------------------------")

#Generating a scatter plot with last row cols
last_two_cols = dataset_random_15.iloc[:,-2:]
plt.scatter(last_two_cols.iloc[:,0],last_two_cols.iloc[:,1])
plt.xlabel("malfind.protection")
plt.ylabel("svcscan.nservices")
plt.title("Scatter Plot for Last Two Rows")
#plt.show()
print('-----------------------------------------------------------------')

#Generating histogram with numerical attributes
first_three_attributes = dataset_random_15.iloc[:,1:4]
#plt.hist('callbacks.ncallbacks','dlllist.avg_dlls_per_proc','pslit.avg_threads')
first_three_attributes.plot.hist(alpha=0.5, bins=20, figsize=(12, 6), layout=(1, 3), sharex=True, sharey=True)
plt.suptitle('Histograms for First Three Numerical Attributes (Randomly Selected 15 Rows & Class Col excluded)')
#plt.show()
print('-----------------------------------------------------------------')


#Describing the spread and distribution of a numerical attribute
print("Describing the spread and distribution of a numerical attribute - 'dlllist.avg_dlls_per_proc'")
print(dataset_random_15['dlllist.avg_dlls_per_proc'].describe())
print('-----------------------------------------------------------------')

#Use one more display of your liking to visualize the numerical attribute; 
numerical_attribute = dataset_random_15['dlllist.avg_dlls_per_proc']

plt.figure(figsize=(12, 6))

# Box plot
plt.subplot(1, 3, 1)
sns.boxplot(x=numerical_attribute)
plt.title('Box Plot')

# Histogram
plt.subplot(1, 3, 2)
sns.histplot(numerical_attribute, bins=30, kde=True)
plt.title('Histogram')

# Kernel Density Plot
plt.subplot(1, 3, 3)
sns.kdeplot(numerical_attribute, fill=True)
plt.title('Kernel Density Plot')

plt.suptitle('Visualizing Numerical Attribute')
#plt.show()
print('-----------------------------------------------------------------')

# Selecting two numerical attributes for scatter plots
attribute1 = 'callbacks.ncallbacks'
attribute2 = 'dlllist.avg_dlls_per_proc'
class_variable = 'Class'

colors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #Variable colors for different 15 variables

# Create a scatter plot using matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(dataset_random_15[class_variable], dataset_random_15[attribute1], c=colors, cmap='magma', alpha=0.5)
plt.xlabel(class_variable)
plt.ylabel(attribute1)
plt.title(f'Scatter Plot: {attribute1} vs {class_variable}')

plt.subplot(2, 2, 2)
plt.scatter(dataset_random_15[class_variable], dataset_random_15[attribute2], c=colors, cmap='magma', alpha=0.5)
plt.xlabel(class_variable)
plt.ylabel(attribute2)
plt.title(f'Scatter Plot: {attribute2} vs {class_variable}')

# Create a scatter plot using seaborn
plt.subplot(2, 2, 3)
sns.scatterplot(x=class_variable, y=attribute1, hue=class_variable, data=dataset_random_15, palette='viridis', alpha=0.7)
plt.title(f'Scatter Plot(seaborn): {attribute1} vs {class_variable}')

plt.subplot(2, 2, 4)
sns.scatterplot(x=class_variable, y=attribute2, hue=class_variable, data=dataset_random_15, palette='viridis', alpha=1)
plt.title(f'Scatter Plot(seaborn): {attribute2} vs {class_variable}')

plt.tight_layout()
#plt.show()
print('-----------------------------------------------------------------')

first_four_attributes = dataset_random_15.iloc[:,:4]
sns.pairplot(first_four_attributes,hue=class_variable,palette='plasma', diag_kind='hist', markers=["o", "s", "D"])
plt.suptitle("Pairplot for Selected Feature with Class Coloring")
plt.show()
print('-----------------------------------------------------------------')

#Logistic Regression 
X = last_two_cols
y = dataset_random_15[class_variable]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)

y_pred_lr = model_lr.predict(X_test)

print("Logistic Regression Accuracy : ",accuracy_score(y_test,y_pred_lr))
print("Classification Report : ",classification_report(y_test,y_pred_lr))