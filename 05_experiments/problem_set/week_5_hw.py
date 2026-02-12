###################################################################################################################
#Week 5 HW Assignment -- Experiments and RCT's
#Kawain Lo
#SoDA 501
###################################################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.special as sc
import statsmodels.api as sm
import statsmodels.formula.api as smf

###################################################################################################################
##PROMPT: Add a retention-style outcome and estimate its ATE. Extend the pipeline so that, in
#addition to the existing outcomes, you compute a user-level retention / activity measure.

##COPIED OVER FROM DEMO TO SETUP
# Reproducibility seed
np.random.seed(123)

# "Big data" knobs (adjust upward if you want more scale)
n_users = 100000     # number of users in the experiment
n_days  = 14         # number of post-assignment days to log


user_id = np.arange(1, n_users + 1)

platform = np.random.choice(
    ["ios", "android", "web"],
    size=n_users,
    replace=True,
    p=[0.35, 0.35, 0.30]
)

cluster_id = np.random.randint(1, 501, size=n_users)

baseline_activity = np.random.gamma(shape=2.0, scale=2.0, size=n_users)

signup_cohort = np.random.choice(
    ["cohort_A", "cohort_B", "cohort_C"],
    size=n_users,
    replace=True,
    p=[0.40, 0.35, 0.25]
)

users = pd.DataFrame({
    "user_id": user_id,
    "platform": platform,
    "cluster_id": cluster_id,
    "baseline_activity": baseline_activity,
    "signup_cohort": signup_cohort
})

# Pre-treatment metric (placebo outcome) correlated with baseline_activity
users["pre_metric"] = users["baseline_activity"] + np.random.normal(0, 0.5, size=n_users)
##Blocked assignment is when you separate your random sample into specific groups and randomize the sample within those groups (aka blocks)

# Blocking: deciles of baseline activity
users["block"] = pd.qcut(users["baseline_activity"], 10, labels=False) + 1  # 1..10

# Randomize within blocks (50/50)
# (groupby + transform returns aligned vector; no functions defined)
users["treat"] = (
    users.groupby("block")["user_id"]
    .transform(lambda s: (np.random.rand(len(s)) < 0.5).astype(int))
)

assignment = users[[
    "user_id", "treat", "block", "platform", "cluster_id",
    "signup_cohort", "baseline_activity", "pre_metric"
]].copy()

assignment["assignment_date"] = np.datetime64("2026-04-16")
# SAVE assignment table (essential reproducibility artifact)

dt_assign = assignment.copy()
dt_assign["dummy"] = 1

dt_days = pd.DataFrame({"day": np.arange(1, n_days + 1)})
dt_days["dummy"] = 1

logs = dt_assign.merge(dt_days, on="dummy", how="outer").drop(columns=["dummy"])

# Date variable
logs["date"] = logs["assignment_date"] + pd.to_timedelta(logs["day"] - 1, unit="D")

# Day-of-week (Mon=1 ... Sun=7) to match the R logic
import pandas
logs["dow"] = logs["date"].dt.dayofweek + 1

# Logging instrumentation dropout
logs["logged_ok"] = (np.random.rand(len(logs)) < 0.98).astype(int)

# Base click rate (Poisson intensity)
logs["base_rate"] = np.exp(
    -1.2
    + 0.15 * np.log1p(logs["baseline_activity"])
    + 0.05 * (logs["platform"] == "ios").astype(float)
    + 0.03 * (logs["platform"] == "android").astype(float)
    + 0.02 * (logs["dow"].isin([6, 7])).astype(float)
    + 0.01 * logs["day"]
)


# Treatment effect (~5% lift)
logs["click_rate"] = logs["base_rate"] * np.exp(0.05 * logs["treat"])

# Click counts (Poisson)
logs["clicks"] = np.random.poisson(lam=logs["click_rate"].to_numpy())

# Purchase probability (logistic)
# logistic(x) = 1 / (1 + exp(-x))
lin = (
    -5.0
    + 0.08 * logs["clicks"]
    + 0.10 * np.log1p(logs["baseline_activity"])
    + 0.15 * logs["treat"]
    + 0.02 * (logs["dow"].isin([6, 7])).astype(float)
)
logs["purchase_prob"] = sc.expit(lin.to_numpy())

# Purchase (Bernoulli)
logs["purchase"] = (np.random.rand(len(logs)) < logs["purchase_prob"].to_numpy()).astype(int)
logs["active"] = ((logs["clicks"] > 0) | (logs["purchase"] > 0)).astype(int)

logs["active"] = logs["active"].where(logs["logged_ok"] == 1, np.nan)

##HOMEWORK STARTS####################################################################
#STEP ONE: From the user-day logs, create days active = the number of days with active == 1 for each user (ignore missing days).

user = (
    logs.groupby("user_id", as_index=False)
        .agg(days_active=("active", lambda x: (x == 1).sum()))
)

#Create retained any = 1 if days active ≥ 1, else 0.
user["retained"] = (user["days_active"] >= 1).astype(int)


print(user.columns)
print(logs.columns)

logs["retained"] = user["retained"]
logs["days_active"] = user["days_active"]

#STEP TWO: Add both outcomes to the analysis-ready dataset and estimate the ATE using:

user = (
    logs.groupby([
        "user_id", "treat", "block", "platform", "cluster_id",
        "signup_cohort", "baseline_activity", "pre_metric", "retained", "days_active"
    ], as_index=False)
    .agg(
        post_clicks=("clicks", "sum"),
        post_purchases=("purchase", "sum"),
        days_observed=("active", lambda x: x.notna().sum()),
        missing_share=("active", lambda x: x.isna().mean()),
    )
)
user.to_csv("data/processed/hw5_analysis_dataset.csv", index=False)

#(a) difference in means, and

means = user.groupby('treat')[['retained', 'days_active']].mean()

# Compute ATE = treated mean - control mean
ate = means.loc[1] - means.loc[0]

print("ATE estimates:")
print(ate)

ate_simple = pd.DataFrame({
    "hw_outcome": ["days active", "retention"],
    "hw_ate_diff_in_means": [ate['days_active'], ate['retained']]
})

#(b) regression adjustment with factor(block) and cluster-robust SEs clustered at cluster id.
import statsmodels.formula.api as smf


fit_conv = smf.ols(
    "days_active ~ treat + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

fit_pur = smf.ols(
    "retained ~ treat + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

#Save your results as outputs/tables/ate retention.csv
ate_simple.to_csv("outputs/tables/hw5_ate_diff_in_means.csv", index=False)
fit_conv.summary2().tables[1].to_csv("outputs/tables/hw5_regression_daysactive.csv")
fit_pur.summary2().tables[1].to_csv("outputs/tables/hw5_regression_retention.csv")

###################################################################################################################
##PROMPT Simulate noncompliance and compare ITT vs TOT (IV). Modify the experiment so
#that treatment assignment does not always translate into treatment receipt.
###################################################################################################################

##STEP ONE: Create a variable "received" such that:

import numpy as np
import pandas as pd

p = 0.4

#all controls have received = 0,
user['received'] = 0

#treated units have received = 1 with probability p < 1 (choose and report your p value),
treated_mask = user['treat'] == 1

#(optional) let p depend on platform or baseline activity.
#Redefine the outcome generation so that the treatment effect operates through received rather than treat.
n_treated = treated_mask.sum()

user.loc[treated_mask, 'received'] = np.random.binomial(1, p, size=treated_mask.sum())

user['hw_outcome'] = ...

user['hw_outcome'] = baseline_activity + treated_mask * user['received'] + np.random.normal(0, 1, size=n_users)

## STEP TWO: Compute

#(a) ITT: regress the outcome on treat (your original approach),
fit_rec = smf.ols(
    "hw_outcome ~ treat + baseline_activity + pre_metric + C(block)",
    data=user
).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})

print(fit_rec.summary)

fit_rec.summary2().tables[1].to_csv("outputs/tables/hw5_regression_ITT.csv")

#(b) TOT / LATE using IV: treat treat as an instrument for received.
from linearmodels.iv import IV2SLS

iv = IV2SLS.from_formula("hw_outcome ~ 1 + C(block) + [treat ~ received]", data=user).fit()

print(iv.summary)

#Report the ITT and TOT estimates side-by-side and explain (2–5 sentences) why TOT is typically larger in magnitude than ITT in your simulation.
#The ITT coefficient for the "treat" variable is 0.3967, while the TOT coefficient is 1.6410. Both are statistically significant at p<0.01. 
#The TOT coefficient will be larger in magnitude than the ITT because of what it is measuring--while the ITT includes the entire sample of those who were *assigned*
#a treatment, it doesn't mean everyone in the sample actually took the treatment. On the other hand, TOT specifically represents the proportion of the sample
#who was assigned to a treatment AND complied with it. Since there aren't any observations with null effects (due to not taking the treatment) in the TOT regression
#there's nothing to drag down the effect size. 

#Save your results as outputs/tables/itt vs tot.csv.
TOT_table = pd.DataFrame({
    "coef": iv.params,
    "std_err": iv.std_errors,
    "t": iv.tstats,
    "p_value": iv.pvalues
})
TOT_table.to_csv("outputs/tables/itt vs tot.csv")
