# air-pollution-python
Air-Pollution visualization and PM2.5 prediction with Python

The main idea of this project is to display a .csv dataset, by using Matplotlib inside the "Shiny" library for Python.

Here is a breakdown of what each code file does in the project:

# a6_ex1.py:
Preprocessing of the data - in other words:
1 - obtaining and extracting the .zip file so I can get the many .csv files
2 - I use the Pandas library to modify the dataset, by loading the set and also by adding and removing
 features of it.
3 - I save the new .csv file

# a6_ex2.py:
Plotting data, here I display
- a PM2.5 Trend as Line Chart
- a Correlation Matrix (by using a HeatMap)
- a Histogram for PM2.5 Levels

# a6_ex3.py:
DataLoader and Dataset - splitting the data for NN training (80 train/ 10 validation/ 10 test). In addition, I save scalar file with feature columns, which will be used later during training.

# a6_ex4.py:
FNN model - In a very simple form and variation of it:
- 1 single Sequential Layer, with 1 Linear input, ReLU, Dropout and finally Linear output
- Hidden size is 32, dropout = 0.5 and output is 1
- Forward pass is defined here

# a6_ex5.py:
Training the model and plotting the final prediction (IMPORTANT NOTE: I still toy with the parameters and the model itself to improve it):
- For optimizer I use Adam with learning rate 0.001 and weight decay 5e-2, 50 epochs

# a6_ex6.py:
This is just a frontend code with Shiny, in the website I display most of the stuff done in the previous files.

# IMPORTANT REQUIREMENTS:
pandas>=2.0.0
matplotlib>=3.7.0
torch>=2.0.0
scikit-learn>=1.3.0
shiny>=0.6.0
shinywidgets>=0.3.0
