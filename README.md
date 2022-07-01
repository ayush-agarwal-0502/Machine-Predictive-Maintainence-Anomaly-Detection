# Machine-Predictive-Maintainence-Anomaly-Detection
Anomaly Detection deployed on machine data dataset for Predictive Maintenance . 

NOTE : I have also included the code files for predictive maintainence using XGBoost , using LSTMs and using Autoencoders (seq2seq and CNN based ) which I learnt from Nvidia Deep learning course , in this repository . However , these were only included as a reference to readers on what other methods could be used and for the sake of completeness , and are not written by me . The code for multivariate gaussian method is written completely by me :)

* Name - Ayush Agarwal 
* Project - Predictive Maintainence Anomaly Detection 
* Skills - Anomaly Detection, Multivariate Gaussian , Unsupervised Learning , Data Visualisation , Exploratory Data Analysis , Data Science , Machine Learning
* Tools - Google Colab , Jupyter Notebooks , Python , Numpy , Pandas , Matplotlib , Seaborn , Sklearn

## Real Life Application :

This can be deployed in an Industrial Internet of Things (IIoT) System , where the system variables measured throught the sensors are sent to Processing System (ofc with my algorithm inside it ) and can be used to monitor system health and predict need for maintainence in the Mechatronic System . Since Prevention is better than cure , a maintainence before any damage or major inconvenience happens is better , hence our system can reduce maintainence costs thus being a business boon . 

## The Dataset : 

The data was taken from Kaggle site : https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification .

A copy of the dataset has been uploaded on this repository just in case the kaggle one gets changed : https://github.com/ayush-agarwal-0502/Machine-Predictive-Maintainence-Anomaly-Detection/blob/main/predictive_maintenance.csv .

## The Code :

The code is availaible at : https://github.com/ayush-agarwal-0502/Machine-Predictive-Maintainence-Anomaly-Detection/blob/main/anomaly_detector_ML_project.ipynb (in this repository itself ) .

## Aim : 

To build an __Anomaly Detection system__ for __Predictive Maintainence__ work on __Machines data__ using __Multivariate Gaussian fitting and thresholding__ , a simple __unsupervised learning__ technique . 

## Data Preprocessing and Visualisation :

Removing UID , ProductID since they are not useful in prediction . Removing Failure type as here our aim is just to predict failure or not , rather than the type of failure itself .

![image](https://user-images.githubusercontent.com/86561124/174450376-4a6aa6e3-9985-44d9-b24b-a933c121f6e9.png)

Using Ordinal Encoding for Type , as it is categorical data .

![image](https://user-images.githubusercontent.com/86561124/174450395-9d542537-a5af-4565-aae1-b0d7c854e30c.png)

Scaling the dataset for better fitting of the gaussian .

![image](https://user-images.githubusercontent.com/86561124/174450428-ba60ca96-81f3-4d0d-a893-52d8a3e9a810.png)

## Fitting the Gaussian :

![image](https://user-images.githubusercontent.com/86561124/174450471-61c40d48-ce63-446e-8874-a200580063f1.png)
![image](https://user-images.githubusercontent.com/86561124/174450476-5cf7fb5a-f6ae-42c9-8abf-569fc5ac1d41.png)

## Making predictions , deciding threshold :

![image](https://user-images.githubusercontent.com/86561124/174450530-b05e5c14-d254-4edb-b1c1-da160ed525cd.png)

![image](https://user-images.githubusercontent.com/86561124/174450553-60c4b40b-ee53-4e8a-9af3-532fd7ddba18.png)

## Result :

![image](https://user-images.githubusercontent.com/86561124/174450564-5fc44003-f86e-4297-b96e-d66ebf6be711.png)

![image](https://user-images.githubusercontent.com/86561124/174450574-5e8e37c4-7b5a-42cd-b8bd-f3b49e3dfdb8.png)

The Confusion Matrix was satisfactory , and the F1 score of my model was in range of that of competitors (https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/code) who have used advanced techniques such as XGBoost , hence showing that the project was successful . Furthermore , the threshold can be shifted to decrease the number of false negatives . The data is as such that the anomalies and the non anomalies are a bit mixed , and hence we can't expect extreme high accuracies . F-Beta Score can also be considered as a better marking factor here if we wanted to emphasise over false negatives over the cost of false positives . 
