# DisasterResponse
Classification of messages into disaster response relevant categories

## **1. Summary**

The project is the fifth project for Udacity's Data Science for Enterprise nano-degree. This project is putting to use my newly acquired sofware development and data engeneering skills in order to analyze data gathered and compiled during disaster by a company called Figure Eight and build an API that classifier the messages into possible disaster-related categories. 

The project was carried out in three broad steps:
- Building an ETL pipeline that extracts, cleans and save the data into a database
- Building a ML pipeline that defines, trains and tests a model that classifies the messages
- Developing a Web application as a front end for the classification model

## **2. Repository set-up**

The repository is set up in the following way:

 - app
   - templates: folder with html templates provided by Udacity
   - run.py: the python file that executes the web app
    
    
 - data
   - DisasterResponse.db: database created at the end of the ETL script
   - disaster_categories.csv: part of the original data provided by Figure Eight
   - disaster_messages.csv: other half of the disaster data provided by Figure Eight
   - process_data.py: python file that contains the ETL pipeline
    
    
 - models
   - model.pkl: pickle file containing the defined and trained classification model
   - train_classifier.py: python file that contains the ML pipeline

## **3. How to run the script**

Additional Resources used to facilitate the code:
- NLTK natural language processing library
- SQLalchemy SQLlite database library
- Flask web app build
- Plotly visualazation tools
- Sciki-learn ML tools
- Pandas tables management

Please follow the below steps in order to execute the python scripts and the web app
### 3.1 Clone this repository
```https://github.com/thebelljar/DisasterResponse```

### 3.2 Run the ETL pipeline that cleans and stores the data in the database
```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```

### 3.3 Run ML pipeline that trains classifier and saves it as pickle file
```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

### 3.4 Change directory to 'app' and execute
```python run.py```

### 3.5 Go to http://0.0.0.0:3001/

## **4. Acknowledgements**

4.1 the project was developed and hosted by [Udacity] (https://www.udacity.com/)

4.2 The data was gathered and initially cleaned by [Figure Eight] (https://www.figure-eight.com/)



