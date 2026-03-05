###########################################################################################################
#WEEK 08 HOMEWORK 
#KAWAIN LO
#SODA 501
###########################################################################################################
#--------------------
#CONCEPTUAL QUESTION: Please write three to ten sentence explanations for the following question. 
#-------------------
#In the social sciences, what are two advantages of collecting data via an API (instead of web scraping)? 
#What are two limitations or risks (e.g., coverage bias, changing endpoints, rate limits, versioning, missingness)? 
#Explain how you would document API-based data provenance so another researcher could replicate your dataset later.

#Compared to web scraping, API's have one major advantage: structure. While a researcher has to take the extra step of
#constructing a format to "hold" or categorize the outputs of web scraping, which is in its raw form messy and practically
#useless for interpretation, API's already come with a pre-determined structure. This saves a researcher time and energy
#that would have otherwise been spent on organizing and cleaning unstructured data. The other advantage of API is that 
#the host server logs and documents everything a researcher does with their API key--which facilitates the process of
#making your research project replicable. 

#However, API's also come with limitations. The first is accessibility--some databases or websites restrict API keys to paying
#customers (like X), and any data pulled through an API request only contains as much as the owner wants to share with the public. 
#This means that crucial information needed for a specific research question may be missing. API pull requests are also subject to 
#rate limits--a running script may be interrupted halfway through a data pulling process because the system has  
#flagged you for exceeding the limit. If you are flagged enough times, you could be banned from accessing the data altogether.

#To ensure an API-based project is fully replicable, thorough documentation is absolutely necessary. Every action should be logged; 
#any failures, changes, or skips need to be recorded; endpoints, parameters, authentication method, date of collection,
#known limitations/failure instances, etc. should be declared. In worst-case scenarios where the website/database becomes inaccessible
#altogether, having the raw data available for sharing can also be useful for replication attempts. 


#########################################################################################################
#EXERCISES: Use the code in the week's code tutorial and the lecture slides to answer the following questions.
#--------------------------------------------
#SETUP
#--------------------------------------------
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
import statsmodels.formula.api as smf

# FRED API wrapper
from fredapi import Fred

# For reading .rds (RDS) files in Python (state-level poll/census data)
import pyreadr|

# For a quick US states choropleth
import plotly.express as px

import os
from dotenv import load_dotenv, dotenv_values
#-----------------------------------------------
#STEP ONE
#{Create an API key and connect to the API.}
#This week you will use the \texttt{fredr} package, which requires a FRED API key.
  #Create a FRED API key.
  #Update the provided script so it reads your key from a local environment variable (recommended) rather than hard-coding it in the file.
  #Confirm your API call works by successfully downloading at least one FRED time series (e.g., unemployment, GDP, or CPI) for election years.
#--------------------------------------------
# loading variables from .env file
load_dotenv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/.env") 
fred_api_key= os.environ["FRED_API_KEY"]
fred = Fred(api_key=fred_api_key)

# Define observation window based on the election years in the vote data
vote_data = pd.read_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/demo/1976-2020-president.csv")
vote_data = vote_data[
    vote_data["party_detailed"].isin(["DEMOCRAT", "REPUBLICAN"])
].copy()

# Drop OTHER and blank candidate entries (mimics R filters)
vote_data = vote_data[
    (~vote_data["candidate"].isin(["OTHER", ""])) & #"`" means "NOT" in python--so filters out
    #candidates that are NOT listed as "OTHER" and keeps them in the dataset
    (vote_data["candidate"].notna())
].copy()

# Compute vote percent
vote_data["vote_pct"] = vote_data["candidatevotes"] / vote_data["totalvotes"]

# Election years used in this dataset
election_years = np.sort(vote_data["year"].unique())

obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

unrate = fred.get_series("UNRATE", observation_start=obs_start, observation_end=obs_end)
unrate = unrate.to_frame(name="unemployment_rate")
unrate.index = pd.to_datetime(unrate.index)
unrate = unrate.resample("QE").mean().reset_index().rename(columns={"index": "date"})
unrate["year"] = unrate["date"].dt.year
unrate["quarter"] = unrate["date"].dt.quarter
unemployment_data = unrate[
    (unrate["year"].isin(election_years)) &
    (unrate["quarter"] <= 2)
][["year", "quarter", "unemployment_rate"]].copy()
#this creates a subset of data in which the observations are categorized into the fiscal quarters Q1 and Q2

gdp = fred.get_series("GDP", observation_start=obs_start, observation_end=obs_end)
gdp = gdp.to_frame(name="gdp")
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.resample("QE").mean().reset_index().rename(columns={"index": "date"})
gdp["year"] = gdp["date"].dt.year
gdp["quarter"] = gdp["date"].dt.quarter
gdp_data = gdp[
    (gdp["year"].isin(election_years)) &
    (gdp["quarter"] <= 2)
][["year", "quarter", "gdp"]].copy()
#restricts observations to those years that occur in election years

cpi = fred.get_series("CPIAUCSL", observation_start=obs_start, observation_end=obs_end)
cpi = cpi.to_frame(name="cpi")
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi.resample("QE").mean().reset_index().rename(columns={"index": "date"})
cpi["year"] = cpi["date"].dt.year
cpi["quarter"] = cpi["date"].dt.quarter
cpi_data = cpi[
    (cpi["year"].isin(election_years)) &
    (cpi["quarter"] <= 2)
][["year", "quarter", "cpi"]].copy()

inflation_data = cpi_data.sort_values(["year", "quarter"]).copy()
inflation_data["inflation_rate"] = (
    (inflation_data["cpi"] / inflation_data["cpi"].shift(2) - 1) * 100
)

# Merging all 3 economic datasets into one long-format table keyed by (year, quarter)
combined_long = (
    unemployment_data
    .merge(gdp_data, on=["year", "quarter"], how="outer")
    .merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer")
    .sort_values(["year", "quarter"])
)

# Pivot wider like R pivot_wider(names_from=quarter, values_from=c(...), names_sep="_Q")
combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi"],
    aggfunc="first"
)

# Flatten column names to match the R naming style, e.g. unemployment_rate_Q1
combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()

#####################################################################################################################
#STEP TWO:
#{Run the baseline forecaster (replicate).}
#Using the provided code (api_build_model.r), get the baseline pipeline running end-to-end on your machine.
  #If any file paths are absolute, modify them to use a {relative path} that will work inside your course project folder.
  #Produce at least one baseline output showing model predictions on held-out data (e.g., 2020), such as a printed table, 
  #summary statistics, or a map.
#----------------------------------------------
forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()

# Incumbent indicator (hard-coded, sequential assignments like the R mutate/if-else chain)
forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD") & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY") & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD") & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL") & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.") & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.") & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.") & (forecast_data["year"] == 2020), "incumbent"] = 1

# Quarter-to-quarter changes (Q2 - Q1), matching the R code
forecast_data["gdp_change"] = forecast_data["gdp_Q2"] - forecast_data["gdp_Q1"] #change in GDP from Q1 to Q2
forecast_data["cpi_change"] = forecast_data["cpi_Q2"] - forecast_data["cpi_Q1"] #change in CPI from Q1 to Q2
forecast_data["unemploy_change"] = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"] #change in unemployemnt from Q1 to !2

# Split training (pre-2020) vs testing (2020)
forecast_data_training = forecast_data[forecast_data["year"] < 2020].copy()
forecast_data_testing  = forecast_data[forecast_data["year"] == 2020].copy()
#data is split to prevent leakage, aka NOT training your model using "future" data (i.e. anything after 2020)
#you train your model ONLY on past data (before 2020) and use it to predict outcomes after 2020

# Fit the national OLS model
# R: vote_pct ~ incumbent * unemploy_change + party_detailed + poly(year, 2, raw = T)
# Python: use year + year^2 explicitly
train_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=forecast_data_training
).fit()
#model calculates an interaction between incumbent president + unemployment rate, year is included as control
forecast_data_training["pred_vote"] = train_ols.predict(forecast_data_training)
print(forecast_data_training[["vote_pct", "pred_vote"]].head(20))

test_pred = train_ols.predict(forecast_data_testing)
print("\n2020 test predictions (first few):")
print(test_pred.head())

#creating table for baseline output
import pandas as pd

headers = ["vote_pct", "pred_vote"]
a = [[0.557273, 0.471090], [0.426149, 0.525077], [0.579046, 0.525077], [0.356531, 0.471090],
     [0.563661, 0.525077], [0.398000, 0.471090], [0.649617, 0.471090], [0.349043, 0.525077],
     [0.497483, 0.525077], [0.479548, 0.471090], [0.540278, 0.525077], [0.426099, 0.471090],
     [0.518814, 0.525077], [0.467337, 0.471090], [0.519691, 0.471090], [0.465876, 0.525077],
     [0.816312, 0.471090], [0.165095, 0.525077], [0.519261, 0.471090], [0.466424, 0.525077]]
df = pd.DataFrame(a, columns=headers)
print(df)
df.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/base_model_train_vs_test.csv", index=False)
##################################################################################################################################
#STEP THREE: {Build a better out-of-sample forecaster.}
#Starting from the baseline, improve out-of-sample performance for election results that are not used to fit the model.
    #Define an explicit out-of-sample design:
      #Hold out the most recent election year in the dataset (e.g., train on years < 2020, test on 2020), or

    #Implement {at least two} model improvements. I chose:
      #adding additional FRED indicators (via the API) and justifying them, 
      #changing the functional form (interactions, nonlinear terms),
     
pinc = fred.get_series("PCE", observation_start=obs_start, observation_end=obs_end)
pinc = pinc.to_frame(name="personal_inc")
pinc.index = pd.to_datetime(pinc.index)
pinc = pinc.resample("QE").mean().reset_index().rename(columns={"index": "date"})
pinc["year"] = pinc["date"].dt.year
pinc["quarter"] = pinc["date"].dt.quarter

pinc_data = pinc[
    (cpi["year"].isin(election_years)) &
    (cpi["quarter"] <= 2)
][["year", "quarter", "personal_inc"]].copy()

#added personal disposable income, which represents consumer purchasing power and perception of 
#financial security among voters

sp = fred.get_series("SP500", observation_start=obs_start, observation_end=obs_end)
sp = sp.to_frame(name="sp500")
sp.index = sp.to_datetime(sp.index)
sp = sp.resample("QE").mean().reset_index().rename(columns={"index": "date"})
sp["year"] = sp["date"].dt.year
sp["quarter"] = sp["date"].dt.quarter

sp_data = sp[
    (sp["year"].isin(election_years)) &
    (sp["quarter"] <= 2)
][["year", "quarter", "sp500"]].copy()
#added S&P500 for measure of market productivity and stability. elections run on big money and investors' risk assessments

combined_long = (
    unemployment_data
    .merge(gdp_data, on=["year", "quarter"], how="outer")
    .merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer")
    .merge(pinc_data[["year", "quarter", "personal_inc"]], on=["year", "quarter"], how="outer")
    .merge(sp_data[["year", "quarter", "sp500"]], on=["year", "quarter"], how="outer")).sort_values(["year", "quarter"]
    )

combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi", "personal_inc", "sp500"],
    aggfunc="first"
)

combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()

forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()


# Incumbent indicator (hard-coded, sequential assignments like the R mutate/if-else chain)
forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD") & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY") & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD") & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL") & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.") & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.") & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.") & (forecast_data["year"] == 2020), "incumbent"] = 1

# Quarter-to-quarter changes (Q2 - Q1), matching the R code
forecast_data["gdp_change"] = forecast_data["gdp_Q2"] - forecast_data["gdp_Q1"] #change in GDP from Q1 to Q2
forecast_data["cpi_change"] = forecast_data["cpi_Q2"] - forecast_data["cpi_Q1"] #change in CPI from Q1 to Q2
forecast_data["unemploy_change"] = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"] #change in unemployemnt from Q1 to !2
forecast_data["personal_inc"] = forecast_data["personal_inc_Q2"] - forecast_data["personal_inc_Q1"]
forecast_data["sp500"] = forecast_data["sp500_Q2"] - forecast_data["sp500_Q1"]

forecast_data_training2 = forecast_data[forecast_data["year"] < 2020].copy()
forecast_data_testing2  = forecast_data[forecast_data["year"] == 2020].copy()

train_ols2 = smf.ols(
    "vote_pct ~ incumbent * gdp_change + personal_inc + cpi_change + unemploy_change + C(party_detailed) + year + I(year**2)",
    data=forecast_data_training2
).fit()

set(forecast_data_training2["party_detailed"])
forecast_data_training2[["incumbent","gdp_change", "personal_inc", "cpi_change", "sp500"]].isna().sum()
#dropped sp500 from model after realizing it has a lot of missingness

forecast_data_training2["pred_vote"] = train_ols2.predict(forecast_data_training2)
print(forecast_data_training2[["vote_pct", "pred_vote"]].head(20))

test_pred2 = train_ols2.predict(forecast_data_testing2)
print("\n2020 test predictions (first few):")
print(test_pred2.head())


#creating output table
header2 = ["new_vote_pct", "new_pred_vote"]
b = [[0.557273, 0.468689], [0.426149, 0.467965], [0.579046, 0.467965], [0.356531, 0.468689],
     [0.563661, 0.467965], [0.398000, 0.468689], [0.649617, 0.468689], [0.349043, 0.467965],
     [0.497483, 0.467965], [0.479548, 0.468689], [0.540278, 0.467965], [0.426099, 0.468689],
     [0.518814, 0.467965], [0.467337, 0.468689], [0.519691, 0.468689], [0.465876, 0.467965],
     [0.816312, 0.468689], [0.165095, 0.467965], [0.519261, 0.468689], [0.466424, 0.467965]]
df2 = pd.DataFrame(b, columns=header2)
print(df2)
df2.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/new_model_train_vs_test.csv", index=False)

#--------------------------------------------
#Report performance {out-of-sample} using at least one quantitative metric 
#(e.g., MAE, RMSE) and compare it to the baseline.

#performance metrics for baseline model
y_true = forecast_data_training["vote_pct"]
y_pred = train_ols.predict(forecast_data_training)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- Baseline Election Forecasting Model out-of-sample performance ---")
print("MAE: ", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R^2: ", round(r2, 4))

metrics0 = pd.DataFrame(
    {"model": ["Modified Election Forecast"], "mae": [mae], "rmse": [rmse], "r2": [r2]}
)
metrics0.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/modified_forecast_metrics.csv", index=False)


#performance metrics for modified/new model

y_true2 = forecast_data_training2["vote_pct"]
y_pred2 = train_ols.predict(forecast_data_training2)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_2 = mean_absolute_error(y_true2, y_pred2)
rmse_2 = np.sqrt(mean_squared_error(y_true2, y_pred2))
r2_2 = r2_score(y_true2, y_pred2)

print("\n--- Modified Election Forecasting Model out-of-sample performance ---")
print("MAE: ", round(mae_2, 4))
print("RMSE:", round(rmse_2, 4))
print("R^2: ", round(r2_2, 4))

metrics = pd.DataFrame(
    {"model": ["Modified Election Forecast"], "mae": [mae_2], "rmse": [rmse_2], "r2": [r2_2]}
)
metrics.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/modified_forecast_metrics.csv", index=False)



#{Communicate the comparison.} Provide:
    #one table that compares baseline vs improved performance (out-of-sample), and
base = pd.read_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/base_model_train_vs_test.csv")
new = pd.read_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/new_model_train_vs_test.csv")

performance_merge = base.merge(new, on="output")
print(performance_merge)
performance_merge.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/performance_comparison.csv", index=False)
    
    #one figure that communicates model fit or errors (e.g., predicted vs actual; error by state; time series comparison).

import pandas as pd


mae = pd.DataFrame({
    "Model": ["Baseline", "New"],
    "mae": [0.0782, 0.0776]
})

rmse = pd.DataFrame({
    "Model": ["Baseline", "New"],
    "rmse": [0.1049, 0.1045]
})

r2 = pd.DataFrame({
    "Model": ["Baseline", "New"],
    "r^2": [0.093, 0.0995]
})

modelfit_comp = mae.merge(rmse, on="Model").merge(r2, on="Model")

print(modelfit_comp)
modelfit_comp.to_csv("C:/Users/karra/Desktop/Coding_work/soda_501/08_api/problem_set/model_metrics_comparison.csv", index=False)
