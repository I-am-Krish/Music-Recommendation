# Music Recommendation System
#### An end to end Machine Learing Pipeline project

## Overview
### This project, titled "Music Recommendation System," is an end-to-end machine learning project aimed at developing a system that can recommend music tracks based on user preferences and listening history. The project leverages a dataset containing various music attributes to train a machine learning model that can predict music preferences.


## Steps followed to build this project

### Step 1: Create a new environment
```
conda create -p venv python==3.10

conda activate venv/
```

### Step 2: Create a requirements.txt file
```
pip install -r requirements.txt
```

### Step 3: Create a requirements.txt file
```
pip install -r requirements.txt
```

### Step 4: Create a setup.py file
```
This is to install the entire project as a package. Additionally, write a function to read the packages from requirements.txt
```

### Step 5: Create a folder `src`
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. The src folder will include another folder with name components will be created. Include __init__.py
also
```

### Step 5.1 Create a folder `components`
```
Include data_ingestion, data_transformation, model trainer, and __init__.py. These components are to be interconnected in future.
```

### Step 5.2 Create a folder called `notebooks`
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
```

### Step 6: Create a folder called `notebook`
```
Create a folder called data and include the dataset. Additionally, create a EDA.ipynb file to do the EDA analysis.
```