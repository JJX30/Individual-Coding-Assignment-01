import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pygal_maps_world.maps import World

import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report


data = pd.read_csv("merged.csv").drop_duplicates(
    subset="Article No.", keep="first")

yearly = {}

for year in data.Year:
    yearly[year] = 0


def display():

    for year in data.Year:
        if (year in yearly):
            yearly[year] += 1
        else:
            yearly[year] = 1

    years = list(yearly.keys())
    count = list(yearly.values())

    plt.bar(years, count, color='maroon',
            width=0.4)
    plt.xlabel("Year")
    plt.ylabel("No. of articles published")
    plt.title("Yearly Publication")
    plt.show()

    # returns number of citations
    for year in yearly:
        yearly[year] = sum(list(data.loc[data['Year'] == year, 'Citation']))

    years = list(yearly.keys())
    citations = list(yearly.values())

    plt.bar(years, citations, color='blue',
            width=0.4)
    plt.xlabel("Year")
    plt.ylabel("No. of Citations")
    plt.title("Yearly Citations")
    plt.show()


def countryPlot():
    merged = pd.read_csv("merged.csv")
    merged.drop(merged[(merged['Country'] == "0")].index, inplace=True)
    country_list = defaultdict(int)

    for countries in merged["Country"]:
        country_list[countries] += 1

    print(country_list)
    worldmap = World()
    worldmap.title = 'Countries'
    worldmap.add('No. of Publications', {
        toCountry("USA"): country_list.get("USA"),
        toCountry("Cyprus"): country_list.get("Cyprus"),
        toCountry("United Kingdom"): country_list.get("United Kingdom"),
        toCountry("United Arab Emirates"): country_list.get("United Arab Emirates"),
        toCountry("Taiwan"): country_list.get("Taiwan"),
        toCountry("Denmark"): country_list.get("Denmark"),
        toCountry("Canada"): country_list.get("Canada"),
        toCountry("Spain"): country_list.get("Spain"),
        toCountry("China"): country_list.get("China"),
        toCountry("New Zealand"): country_list.get("New Zealand"),
        toCountry("Chile"): country_list.get("Chile"),
        toCountry("Italy"): country_list.get("Italy"),
        toCountry("Australia"): country_list.get("Australia"),
        toCountry("Israel"): country_list.get("Israel"),
        toCountry("France"): country_list.get("France"),
        toCountry("Germany"): country_list.get("Germany"),
        toCountry("India"): country_list.get("India"),
        toCountry("Slovakia"): country_list.get("Slovakia"),
        toCountry("Ireland"): country_list.get("Ireland"),
        toCountry("Kyrgyzstan"): country_list.get("Kyrgyzstan"),
        toCountry("Malaysia"): country_list.get("Malaysia"),
        toCountry("Pakistan"): country_list.get("Pakistan"),
        toCountry("Liechtenstein"): country_list.get("Liechtenstein"),
        toCountry("Norway"): country_list.get("Norway"),
        toCountry("Hong Kong"): country_list.get("Hong Kong"),
        toCountry("Korea"): country_list.get("Korea"),
        toCountry("Switzerland"): country_list.get("Switzerland"),
        toCountry("Mexico"): country_list.get("Mexico"),
        toCountry("Ukraine"): country_list.get("Ukraine"),
        toCountry("South Africa"): country_list.get("South Africa"),
        toCountry("Greece"): country_list.get("Greece"),
        toCountry("Russia"): country_list.get("Russia"),
        toCountry("Czech Republic"): country_list.get("Czech Republic"),
        toCountry("Palestine"): country_list.get("Palestine"),
    })
    worldmap.render_to_file('map.svg')


def toCountry(country):
    if (country == "USA"):
        return "us"
    if (country == "Cyprus"):
        return "cy"
    if (country == "United Kingdom"):
        return "gb"
    if (country == "United Arab Emirates"):
        return "ae"
    if (country == "Taiwan"):
        return "tw"
    if (country == "Denmark"):
        return "dk"
    if (country == "Canada"):
        return "ca"
    if (country == "Spain"):
        return "es"
    if(country == "China"):
        return "cn"
    if (country == "New Zealand"):
        return "nz"
    if (country == "Chile"):
        return "cl"
    if (country == "Italy"):
        return "it"
    if (country == "Australia"):
        return "au"
    if (country == "Israel"):
        return "il"
    if (country == "France"):
        return "fr"
    if (country == "Germany"):
        return "de"
    if (country == "India"):
        return "in"
    if (country == "Slovakia"):
        return "sk"
    if (country == "Ireland"):
        return "ie"
    if (country == "Kyrgyzstan"):
        return "kg"
    if (country == "Malaysia"):
        return "my"
    if (country == "Pakistan"):
        return "pk"
    if (country == "Liechtenstein"):
        return "li"
    if (country == "Norway"):
        return "no"
    if (country == "Hong Kong"):
        return "hk"
    if (country == "Korea"):
        return "kr"
    if (country == "Switzerland"):
        return "ch"
    if (country == "Mexico"):
        return "mx"
    if (country == "Ukraine"):
        return "ua"
    if (country == "South Africa"):
        return "za"
    if (country == "Greece"):
        return "gr"
    if (country == "Russia"):
        return "ru"
    if (country == "Czech Republic"):
        return "cz"
    if (country == "Palestine"):
        return "ps"


def regression():
    file = pd.read_csv("data.csv").fillna(0)

    LabelEncoder().fit(file['Purchase'])
    file['Purchase'] = LabelEncoder().transform(file['Purchase'])
    LabelEncoder().fit(file['Gender'])
    file['Gender'] = LabelEncoder().transform(file['Gender'])

    result = file.corr(method='pearson')['SUS'].sort_values()
    print(result)

    y = file['SUS']
    x = file.drop(columns='SUS')

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    print(model.summary())

    x = file.drop(columns='SUS')
    y = file['SUS']

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    y_train_pred = LinearRegression().fit(x_train, y_train).predict(x_train)
    y_test_tred = LinearRegression().fit(x_train, y_train).predict(x_test)

    print("The R square score of linear regression model is: ",
          LinearRegression().fit(x_train, y_train).score(x_test, y_test))

    quad = PolynomialFeatures(degree=2)

    x_quad = quad.fit_transform(x)

    X_train, X_test, Y_train, Y_test = train_test_split(
        x_quad, y, random_state=0)

    # plots the curved line

    Y_train_pred = LinearRegression().fit(X_train, Y_train).predict(X_train)
    Y_test_pred = LinearRegression().fit(X_train, Y_train).predict(X_test)

    print("The R square score of 2-order polynomial regression model is: ",
          LinearRegression().fit(X_train, Y_train).score(X_test, Y_test))


def classification():
    file = pd.read_csv("data.csv").fillna(0)

    # fills missing values with mean valye of column
    file['Purchase'] = file['Purchase'].fillna(file['Purchase'].mean())
    file['SUS'] = file['SUS'].fillna(file['SUS'].file())
    file['Duration'] = file['Duration'].fillna(file['Duration'].mean())
    file['ASR_Error'] = file['ASR_Error'].fillna(file['ASR_Error'].mean())
    file['Gender'] = file['Gender'].fillna(file['Gender'].mean())
    file['Intent_Error'] = file['Intent_Error'].fillna(
        file['Intent_Error'].mean())

    # setting data to be inserted in model
    y = file['Purchase'].to_numpy()
    x = file.drop('Purchase', axis=1).to_numpy()
    # scales for normal distribution
    scaled_X = StandardScaler().fit_transform(x)  # stored

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.4)  # split

    # oversampling method to solve imbalance problems
    oversample = SMOTE()
    over_sampled_X_train, over_sampled_y_train = oversample.fit_resample(
        X_train, y_train)

    # sets up the classification to be tested
    LogisticRegression().fit(over_sampled_X_train, over_sampled_y_train)
    SVC(probability=True).fit(over_sampled_X_train, over_sampled_y_train)
    GaussianNB().fit(over_sampled_X_train, over_sampled_y_train)
    RandomForestClassifier().fit(over_sampled_X_train, over_sampled_y_train)

    #  model stored in an array
    y_lc_predicted = LogisticRegression().predict(X_test)
    y_lc_pred_proba = LogisticRegression().predict_proba(X_test)

    y_svc_predicted = SVC(probability=True).predict(X_test)
    y_svc_pred_proba = SVC(probability=True).predict_proba(X_test)

    y_nbc_predicted = GaussianNB().predict(X_test)
    y_nbc_pred_proba = GaussianNB().predict_proba(X_test)

    y_rfc_predicted = RandomForestClassifier().predict(X_test)
    y_rfc_pred_proba = RandomForestClassifier().predict_proba(X_test)

    # prints out in a table
    print('Linear Regression: \n', classification_report(y_test, y_lc_predicted))
    print('SVC: \n', classification_report(y_test, y_svc_predicted))
    print('Naive Bayes: \n', classification_report(y_test, y_nbc_predicted))
    print('Random Forest: \n', classification_report(y_test, y_rfc_predicted))

    models = ['Logistic Regression', 'Support Vector Machine',
              'Naive Bayes Classifier', 'Random Forest Classifier']
    predictions = [y_lc_predicted, y_svc_predicted,
                   y_nbc_predicted, y_rfc_predicted]
    pred_probabilities = [y_lc_pred_proba,
                          y_svc_pred_proba, y_nbc_pred_proba, y_rfc_pred_proba]

    for model, prediction, pred_proba in zip(models, predictions, pred_probabilities):
        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_test.ravel(), prediction))
        disp.plot(
            include_values=True,
            cmap='gray',
            colorbar=False
        )
        disp.ax_.set_title(f"{model} Confusion Matrix")

    plt.figure(figsize=(30, 15))
    plt.suptitle("ROC Curves")
    plot_index = 1

    # plots ROC curve
    for model, prediction, prob in zip(models, predictions, pred_probabilities):
        fpr, tpr = roc_curve(y_test, prob[:, 1])
        auc_score = auc(fpr, tpr)
        plt.subplot(3, 2, plot_index)
        plt.plot(fpr, tpr, 'r', label='ROC curve')
        plt.title(f'Roc Curve - {model} - [AUC - {auc_score}]', fontsize=14)
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        plt.legend()
        plot_index += 1
    plt.show()
