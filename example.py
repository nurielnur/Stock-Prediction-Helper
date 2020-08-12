###################################################################################
# This is an introductory and simple example of how to use Stock Prediction Helper in order to process data,
# add new features to stock market data and make a prediction of the data.


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import stock_prediction_helper as sph
from sklearn.metrics import classification_report


df = pd.read_csv("DATA/Example.csv")

# Changing str to float if exists.
df = sph.change_punc(df, ["Price", "Open", "High", "Low"])
df["Vol."] = df["Vol."].apply(lambda x: sph.change_volume(x))


# Dropping unnecessary data and reversing data in order to make it in ascending order in terms of time.
df = df.drop(["Open", "High", "Low", "Change %"], axis=1)
df = df.iloc[::-1].reset_index(drop=True)

# Adding 9, 21 and 50 days of Moving Averages. Any other choise for period of time can be added.
df["MovA_9"] = sph.calc_mova(df["Price"], 9)
df["MovA_21"] = sph.calc_mova(df["Price"], 21)
df["MovA_50"] = sph.calc_mova(df["Price"], 50)

# Adding 9, 21 and 50 days of Exponential Moving Averages. Any other choise for period of time can be added.
df["EMA_9"] = sph.calc_ema(df["Price"], 9)
df["EMA_12"] = sph.calc_ema(df["Price"], 12)
df["EMA_26"] = sph.calc_ema(df["Price"], 26)
df["EMA_21"] = sph.calc_ema(df["Price"], 21)
df["EMA_50"] = sph.calc_ema(df["Price"], 50)

# Adding Moving Average Convergence Divergence.
df["MACD_Blueline"] = sph.calc_macd_blueline(df["Price"], 12, 26)
df["MACD_Redline"] = sph.calc_macd_redline(df["MACD_Blueline"].values, 26, 9)

# Adding Label to Dataframe. Label is either 1 if its price is higher than profit
# in 10 consecutive period of time or zero if it is not.
df["Label"] = sph.is_profitable(df["Price"], 10, profit=1.05)

df2 = df.dropna(axis=0)
df2.to_csv("PROCESSED/processed_data.csv")


# Transposing dataframe for every 10 days of data as a single input to ML Model.
df_final = pd.DataFrame(sph.transpose_data_for_train(
    df2, "Label", 10, ["Date", "Label"]))
df_final.to_csv("PROCESSED/final_data.csv")


def split_data(df, target, percent=0.85):
    """
    Returns train and test data as splitted by the delimeter point. 

    Splitting time series data with no shuffling and returning train and test data.  

    Parameters
    ---------
    df : DataFrame
        Dataframe to split.

    target : str
        Target column in the DataFrame.

    percent: float
        Delimeter point to split data (Should be between 0 and 1).
    """

    length = int(len(df)*percent)

    train_df = df.iloc[:length]
    test_df = df.iloc[length:]

    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]

    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    return (X_train, y_train, X_test, y_test)


# Splitting data frame to train and test. Last Column of the dataframe is label.
X_train, y_train, X_test, y_test = split_data(
    df_final, df_final.columns[-1], 0.8)

# Just trying out a Random Forest Model with. Any other choice of model can be tested.
rf_model = RandomForestClassifier(n_estimators=1000, max_features=0.2)
rf_model.fit(X_train, y_train)


#############################################
# Classifiation report of the model
# pred = rf_model.predict(X_test)
# print(classification_report(y_test, pred))
#############################################


# If the prediction for the last 10 consecutive days(test) in the data is 1,
# then it means price of the stock will be higher than last days value
# with at least %5 in upcoming 10 days(If precision of the model is high).
test = sph.transpose_data_for_test(df.iloc[-10:], 10, ["Date"])
pred = rf_model.predict(X_test)

print(classification_report(y_test, pred))
