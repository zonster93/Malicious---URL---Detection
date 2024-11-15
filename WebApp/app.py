#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
from feature import generate_data_set
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV

data = pd.read_csv("phishing.csv")
#droping index column

data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature
data=data.drop(columns=["PageRank"],axis=1)
X = data.drop(["class"],axis =1)
y = data["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

clf_rfecv_model = gbc.fit(X_train, y_train)
gbc.fit(X_train, y_train)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        inactive_df = pd.read_csv(r"inactive.csv")
        # Take user input
        search_input = url

        # Check if the input exists in the 'column_name' column
        filtered_rows = inactive_df[inactive_df['subject'] == search_input]

        if not filtered_rows.empty:
            # Assuming the URL is in the 'url_column' column
            url_value = filtered_rows.iloc[0]['subject']
            if pd.notna(url_value):
                return render_template('contact.html',url=url,res=2 )
            else:
                print("URL not present")
        else:
            x = np.array(generate_data_set(url)).reshape(1,29) 
            y_pred =gbc.predict(x)[0]
            #1 is safe       
            #-1 is unsafe
            y_pro_phishing = gbc.predict_proba(x)[0,0]
            y_pro_non_phishing = gbc.predict_proba(x)[0,1]
            # if(y_pred ==1 ):
            pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
            return render_template('contact.html',xx =round(y_pro_non_phishing,2),url=url,res=1 )

    return render_template("contact.html", xx =-1,res=1)


if __name__ == "__main__":
    app.run(debug=True)