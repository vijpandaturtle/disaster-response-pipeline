# Disaster Response Pipeline Project
The aim of the project is to classify the different types of messages received during disasters into specific categories using classical machine learning models. This way it becomes easy to segregate the messages into correct categories and send it across to different teams on-site so that they can get help the victims in an efficient manner. 

### Dependencies 
1. NLTK
2. Numpy 
3. Pandas 
4. SQLAlchemy
5. Scikit-learn
6. Regex 
7. Pickle 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
