# AI_Text_Prediction

An AI-text detection application to predict AI-generated text using machine learning models such as SVM, Decision-Trees and AdaBoost Classifiers. The models were deployed using Streamlit.

To use the application:

Visit: https://aitextprediction-kvotlzlbq6hprve9jbgcej.streamlit.app/

To run locally and perform any changes:

1. Download the files from this repository.
   
2. Download all the dependencies from the requirements.txt file. You can do this directly or by creating a python virtual environment.
3. To create the virtual environment, go into the app and tyoe:

   python -m venv your_file_name
   
5. Now, your virtual envirnoment should be set up.
   Activate using,

   your_file_name\Scripts\activate

6. Now cd into the folder and run:

   pip install -r requirements.txt

   to download all the dependencies.
   
8. Now you can run the streamlit app with:

   streamlit run app.py

   **Make sure all the required dependencies are downloaded**

9. The machine learning algorithms are in the notebooks file. You can use the provided pkl files to run the models or change the models in
   the ipynb file and export them
