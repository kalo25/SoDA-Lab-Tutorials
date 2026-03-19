###############################################################################
#SODA WEEK 09 HW: DATA QUALITY AND MEASUREMENT ERROR
#Kawain Lo
################################################################################
#Step 1: Run the demo script end-to-end and confirm it writes outputs to the correct folders
#CONFIRMED

###########################################################
#SETUP
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(123)
n = 5000
# True confounder
x_true = np.random.normal(loc=0.0, scale=1.0, size=n)
# Treatment assignment correlated with x_true (logistic link)
# (This creates confounding: D is not independent of x_true.)
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n) #d refers to non-random assignment of treatment
# True outcome model (based on causal inference model)
tau = 1.0     # true effect of D on Y (confounder effect)
beta = 1.0    # effect of X_true on Y (focal effect)
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n) #generating random noise
y = tau * d + beta * x_true + eps_y #equation for generating the data and its trend
#for the base model starting coefficient is tau
# Placebo outcome (negative control outcome): NOT affected by D by construction
eps_pl = np.random.normal(loc=0.0, scale=1.0, size=n)
y_placebo = 0.0 * d + beta * x_true + eps_pl #placebo outcome has a diff starting coefficient (0) 
df_base1 = pd.DataFrame(
    {
        "y": y, #assigns name/label to the q values defined on line 82
        "y_placebo": y_placebo,
        "d": d,
        "x_true": x_true,
    }
)
sigma_u_grid = [0.0, 0.2, 0.5, 1.0, 2.0] #referenced in chunk starting on line 133
R = 30  # repetitions (to see variability from 30 measurement error draws)
# Fix a "validation sample" index set (20% of observations)
validation_share = 0.20
validation_idx = np.random.choice(np.arange(n), size=int(validation_share * n), replace=False)
is_validation = np.zeros(n, dtype=bool)
is_validation[validation_idx] = True

rows1 = []
for sigma_u in sigma_u_grid: #assigns values to each named array below
    tau_oracle_list = []
    tau_naive_list = []
    tau_cal_list = []
    tau_placebo_list = []
    beta_oracle_list = [] 
    beta_naive_list = []
    beta_cal_list = []
    for r in range(R): #r refers to the 30 draws on line 121
        # Draw measurement error and observed covariate
        u = np.random.normal(loc=0.0, scale=sigma_u, size=n) #sigma_u values become randomized b/c of the function np.random.normal
        x_obs = x_true + u
        # Build design matrices with intercept column
        ones = np.ones(n) #assigns the number 1 as an intercept for every model in the following lines
        # (A) Oracle regression: y ~ 1 + d + x_true
        X_oracle = np.column_stack([ones, d, x_true])
        coef_oracle, _, _, _ = np.linalg.lstsq(X_oracle, y, rcond=None)
        # coef_oracle: [intercept, d, x_true]
        tau_oracle_list.append(coef_oracle[1])
        beta_oracle_list.append(coef_oracle[2])
        # (B) Naive regression: y ~ 1 + d + x_obs
        X_naive = np.column_stack([ones, d, x_obs])
        coef_naive, _, _, _ = np.linalg.lstsq(X_naive, y, rcond=None)
        tau_naive_list.append(coef_naive[1])
        beta_naive_list.append(coef_naive[2])
        # (C) Regression calibration (validation subsample):
        #     estimate x_true ~ x_obs on validation sample, predict x_hat for all.
        ones_val = np.ones(int(validation_share * n))
        X_cal_val = np.column_stack([ones_val, x_obs[is_validation]])
        coef_cal, _, _, _ = np.linalg.lstsq(X_cal_val, x_true[is_validation], rcond=None)
        x_hat = coef_cal[0] + coef_cal[1] * x_obs
        X_calibrated = np.column_stack([ones, d, x_hat])
        coef_calibrated, _, _, _ = np.linalg.lstsq(X_calibrated, y, rcond=None)
        tau_cal_list.append(coef_calibrated[1])
        beta_cal_list.append(coef_calibrated[2])
        # Outcome placebo: y_placebo ~ 1 + d + x_obs
        X_placebo = np.column_stack([ones, d, x_obs])
        coef_placebo, _, _, _ = np.linalg.lstsq(X_placebo, y_placebo, rcond=None)
        tau_placebo_list.append(coef_placebo[1])
    # Summaries per sigma_u
    rows1.append(
        {
            "sigma_u": sigma_u,
            "tau_true": tau,
            "tau_oracle_mean": float(np.mean(tau_oracle_list)),
            "tau_naive_mean": float(np.mean(tau_naive_list)),
            "tau_cal_mean": float(np.mean(tau_cal_list)),
            "tau_placebo_mean": float(np.mean(tau_placebo_list)),
            "tau_oracle_q025": float(np.quantile(tau_oracle_list, 0.025)),
            "tau_oracle_q975": float(np.quantile(tau_oracle_list, 0.975)),
            "tau_naive_q025": float(np.quantile(tau_naive_list, 0.025)),
            "tau_naive_q975": float(np.quantile(tau_naive_list, 0.975)),
            "tau_cal_q025": float(np.quantile(tau_cal_list, 0.025)),
            "tau_cal_q975": float(np.quantile(tau_cal_list, 0.975)),
            "beta_true": beta,
            "beta_oracle_mean": float(np.mean(beta_oracle_list)),
            "beta_naive_mean": float(np.mean(beta_naive_list)),
            "beta_cal_mean": float(np.mean(beta_cal_list)),
        }
    )
    print(f"  done sigma_u={sigma_u}")

#using outputs/measurement.error.results.csv, make a clean table that reports for each sigma_u
    #tau_oracle_mean, tau_naive_mean, tau_cal_mean, 
    #beta_oracle_mean, beta_naive_mean, beta_cal_mean

results1 = pd.DataFrame(rows1)
print("\n--- Summary (means) ---")
print(results1[["sigma_u", "tau_oracle_mean", "tau_naive_mean", "tau_cal_mean", "beta_oracle_mean",
               "beta_naive_mean", "beta_cal_mean"]].to_string(index=False))

results1.to_csv("09_data_quality/problem_set/HW_measurement_error_results.csv", index=False)
##############################
#PROMPT
#################################
#in 6-10 sentences, interpret the figures:
    #measurement_error_tau_vs_sigma.png (Figure 1)
    #measurement_error_beta_vs_sigma.png (Figure 2)
#your interpretation must address:
    #how and why the naive estimate of the treatment effect changes as {sigma_u} increases,
    #attenuation of the confounder coefficient (why the estimated confounder effect shrinks), and
    #the difference between the ``oracle'' and ``naive'' estimands in this simulation.
#################################
###PLEASE NOTE-----
#I ran the demo code without editing the script in any way, and the graph for (Figure 1) measurement_error_tau_vs_sigma.png
#just does not show the trend for "naive" estimates. I checked and the estimates exist. I tried generating
#a graph of just the "naive" estimates alone and that didn't work either. I'm not sure why this is happening.
#-------
#RESPONSE BELOW:
#----------------------------------------------------------
#Tau represents the effect of the focal independent variable (treatment effect) on the outcome variable; 
#conversely, beta represents the effect of the confounding variable (x_true, an unobserved variable) on the outcome variable.
#sigma_u represents random error (in this case, it has been assigned known values)

#The oracle model is best at capturing the full effect of both the treatment and confounding vars
#on the outcome--this is why its plotted values closely follow the baseline ("true tau") in both figures.

#The naive model mimics real-life conditions, in which variable observations are contaminated with
#random error and noise. In Figure 2, which shows the coefficient for the confounder variable, the naive model
#the estimates for the naive model sharply decrease with every increase in measurement error.
#I assume that the estimated confounder effect has an inverse relationship with random error b/c the
#amount of noise generated by the random error will artificially inflate the effect of D (the treatment variable),
#which in turn obscures the effects of the confounding variable.

#The calibrated model uses a fine-tuned variable (derived from comparing and refining the estimates of 
#the treatment var and the confounding var) to predict the outcome. Therefore, any effects of D (treatment var) will be 
#more accurate to the ground truth.

#In Figure 1, the graphed estimates for the placebo model fall far below that of the other models.
#This is also reasonable, since the placebo model is meant to represent a null treatment effect (and
#the increase in estimates is correlated with the increase in random error, as explained above).
###############################################################################################################
#Step 2: Validation subsample and regression calibration
######################################
#change validation share to 3 values (0.05, 0.20, 0.50) by editing [validation_share]

validation_share_grid = [0.05, 0.20, 0.50]

rows2 = []
#hold [sigma_u] fixed at ONE moderate value (pick any number) and rerun the simulation for each validation-share setting
sigma_u_grid = [1.0] 
R = 30

for sigma_u in sigma_u_grid:
    tau_oracle_list = []
    tau_naive_list = []
    tau_placebo_list = []
    beta_oracle_list = []
    beta_naive_list = []

    for r in range(R):
        u = np.random.normal(0, sigma_u, n)
        x_obs = x_true + u
        ones = np.ones(n)

        #oracle
        X_oracle = np.column_stack([ones, d, x_true])
        coef_oracle, _, _, _ = np.linalg.lstsq(X_oracle, y, rcond=None)
        tau_oracle_list.append(coef_oracle[1])
        beta_oracle_list.append(coef_oracle[2])

        #naive
        X_naive = np.column_stack([ones, d, x_obs])
        coef_naive, _, _, _ = np.linalg.lstsq(X_naive, y, rcond=None)
        tau_naive_list.append(coef_naive[1])
        beta_naive_list.append(coef_naive[2])

        #placebo
        coef_placebo, _, _, _ = np.linalg.lstsq(X_naive, y_placebo, rcond=None)
        tau_placebo_list.append(coef_placebo[1])

    #validation
    for validation_share in validation_share_grid:
        tau_cal_list = []
        beta_cal_list = []

        for r in range(R):
            u = np.random.normal(0, sigma_u, n)
            x_obs = x_true + u
            ones = np.ones(n)

            #validation mask
            validation_idx = np.random.choice(np.arange(n), size=int(validation_share * n), replace=False)
            is_validation = np.zeros(n, dtype=bool)
            is_validation[validation_idx] = True

            ones_val = np.ones(is_validation.sum())
            X_cal_val = np.column_stack([ones_val, x_obs[is_validation]])
            coef_cal, _, _, _ = np.linalg.lstsq(X_cal_val, x_true[is_validation], rcond=None)
            x_hat = coef_cal[0] + coef_cal[1] * x_obs

            #calibrated
            X_calibrated = np.column_stack([ones, d, x_hat])
            coef_calibrated, _, _, _ = np.linalg.lstsq(X_calibrated, y, rcond=None)
            tau_cal_list.append(coef_calibrated[1])
            beta_cal_list.append(coef_calibrated[2])

        rows2.append({
            "sigma_u": sigma_u,
            "validation_share": validation_share,
            "tau_cal_mean": float(np.mean(tau_cal_list)),
            "tau_naive_mean": float(np.mean(tau_naive_list)),
            "tau_oracle_mean": float(np.mean(tau_oracle_list)),
            "tau_placebo_mean": float(np.mean(tau_placebo_list)),
            "tau_cal_q025": float(np.quantile(tau_cal_list, 0.025)),
            "tau_cal_q975": float(np.quantile(tau_cal_list, 0.975)),
            "tau_naive_q025": float(np.quantile(tau_naive_list, 0.025)),
            "tau_naive_q975": float(np.quantile(tau_naive_list, 0.975)),
            "tau_oracle_q025": float(np.quantile(tau_oracle_list, 0.025)),
            "tau_oracle_q975": float(np.quantile(tau_oracle_list, 0.975)),
            "beta_cal_mean": float(np.mean(beta_cal_list)),
            "beta_naive_mean": float(np.mean(beta_naive_list)),
            "beta_oracle_mean": float(np.mean(beta_oracle_list)),
        })
    
#report for each validation share the mean calibrated treatment estimate (tau_cal_mean) and compare it to the naive estimate (tau_naive_mean)
results_hw = pd.DataFrame(rows2)
print("\n--- HW Summary (means) ---")
print(results_hw[['sigma_u', 'validation_share', 'tau_cal_mean', 'tau_naive_mean']])

results_hw.to_csv("09_data_quality/problem_set/HW_val3_measurement_error_results.csv", index=False)
#############################################
#PROMPT
############################################
#in 6-10 sentences explain what changes as the validation sample grows. Discuss:
    #why calibration can help when the confounder is measured with error,
    #why calibration is not ``magic'' (what assumptions it relies on), and
    #one reason calibration might fail or remain biased in real social data. 
#------------------------
#RESPONSE BELOW:
#------------------------
#the changes in tau_cal_mean estimates are extremely small, and seem to have no clear correlation with the
#validation share. At 20% validation share, the estimate increases from 1.444620 to 1.445265.
#However, at 50% validation share, the estimate drops to 1.442. However, knowing that this estimate corresponds
#with a 50% validation share increases confidence that it is the most accurate estimate out of the three.

#Calibration is a good way to adjust for error because it allows the researcher to refine estimates by training
#a model on observed data collected from multiple sources. I think this is somewhat similar to a Cronbach's alpha--in 
#creating a scale meant to reflect a real-world concept, you need to make sure the variables loaded into the scale
#have an optimal amount of correlation (0.7-0.8). Too low, and the variables need to be discarded because they measure
#entirely different things. Too high, and you don't need a scale at all--any one of the variables can be used to 
#measure the target concept. Along the same lines, calibration provides for a closer, more accurate model of real-world conditions.

#Calibration does come with some assumptions and risks. Since the calibration process relies on already-collected data,
#the researcher needs to be sure that said data is free from misreporting, that the sample itself is randomized
#and representative of the population of interest, and that any measurement error is kept to an absolute minimum.
#If not, the calibrated model risks inflating whatever biases exist in the data it is derived from. This is a real
#and common risk when using big social data, which is full of limitations--unobserved, real events go undetected and don't
#show up on data; active internet or social media users are not representative of the whole population; user behavior on
#social media is constrained by their interactions with the platform algorithm, etc. Calibration
#is not a foolproof remedy for these problems. 

###################################################################################################################
#Step 3: Placebo Tests (outcome placebo and treatment permutation)
#####################################
#Using the demo script:

n = 5000
# True confounder
x_true = np.random.normal(loc=0.0, scale=1.0, size=n)
# Treatment assignment correlated with x_true (logistic link)
# (This creates confounding: D is not independent of x_true.)
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n) #d refers to non-random assignment of treatment
# True outcome model (based on causal inference model)
tau = 1.0     # true effect of D on Y 
beta = 1.0    # effect of X_true on Y 
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n) #generating random noise
y = tau * d + beta * x_true + eps_y #equation for generating the data and its trend
#for the base model starting coefficient is tau
eps_pl = np.random.normal(loc=0.0, scale=1.0, size=n)

y_placebo = 0.0 * d + beta * x_true + eps_pl #placebo outcome has a diff starting coefficient (0) 
df_base = pd.DataFrame(
    {
        "y": y, #assigns name/label to the q values defined on line 82
        "y_placebo": y_placebo,
        "d": d,
        "x_true": x_true,
    }
)
print("\n--- Data preview ---")
print(df_base.head())
print("\nTreatment rate:", round(df_base["d"].mean(), 4))
#Outcome placebo: report the estimated coefficient on {d} in the placebo regression and 
#explain why it should be close to zero.
#-------------------
#RESPONSE BELOW:
#-------------------
#the estimated coefficients on d in the placebo regression all hover between the range of -1.5 and slightly above 1
#this is expected because the placebo regression assumes that there is no correlation whatsoever
#between the focal independent and dependent variables. therefore, there should be no effect at all
#the numbers only vary because of the random noise we introduced when setting up the models

#--------------------------------------------------
#Treatment permutation placebo: run the permutation test and report:
sigma_u_perm = 1.0
u_perm = np.random.normal(loc=0.0, scale=sigma_u_perm, size=n)
x_obs_perm = x_true + u_perm

ones = np.ones(n)
X_obs = np.column_stack([ones, d, x_obs_perm])
coef_obs, _, _, _ = np.linalg.lstsq(X_obs, y, rcond=None)
tau_hat_obs = float(coef_obs[1])

print("\n--- Permutation placebo setup ---")
print("sigma_u used:", sigma_u_perm)
print("Observed tau_hat (naive model):", round(tau_hat_obs, 4))

B = 500
tau_perm = []

for b in range(B):
    d_perm = np.random.permutation(d)
    X_b = np.column_stack([ones, d_perm, x_obs_perm])
    coef_b, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    tau_perm.append(float(coef_b[1]))

tau_perm = np.array(tau_perm)

# Empirical two-sided p-value
p_emp = (1.0 + np.sum(np.abs(tau_perm) >= np.abs(tau_hat_obs))) / (B + 1.0)
print("Empirical p-value (two-sided):", round(p_emp, 4))

    #the observed naive estimate {tau_hat_obs},
        #--------
        #the observed tau_hat estimate is 1.4543 (so an estimate derived from the model
        #that most closely mimics real-life conditions; data collection with random error)
        #--------
    #the empirical two-sided p-value,
        #--------
        # the p-value is around 0.002
        #--------
    #a histogram figure of the permutation distribution with the observed estimate marked.

# Plot permutation distribution + observed line
plt.figure(figsize=(8, 5))
plt.hist(tau_perm, bins=30, alpha=0.8)
plt.axvline(tau_hat_obs, linestyle="--", linewidth=2, label=f"Observed tau_hat = {tau_hat_obs:.3f}")
plt.axvline(-tau_hat_obs, linestyle="--", linewidth=1)
plt.title(f"Treatment permutation placebo (sigma_u={sigma_u_perm})\nEmpirical p-value = {p_emp:.3f}")
plt.xlabel("Coefficient on permuted treatment")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("09_data_quality/demo/figures/HW_permutation_placebo_tau_hist.png", dpi=200)
plt.close()
##############################################
#PROMPT
##############################################
#In 6--10 sentences, interpret what the permutation distribution is telling you. Your interpretation must address:
    #what the null hypothesis is in the permutation test,
    #what it means if the observed estimate is extreme relative to the permutation distribution, and
    #one way placebo tests can detect pipeline problems (e.g., leakage, overfitting, coding errors).
#---------------------------------------------
#RESPONSE BELOW:
#---------------------------------------------
#I might be completely incorrect (and if I am, please let me know!)--I've never heard of permutation tests before.
#Based on my understanding of the model: this randomly varies the "d" (the treatment effect) coefficient
#500 times among the entire sample, then plots each outcome on the distribution graph. This is meant to determine
#whether the treatment actually has an effect on the outcome variable or not--if it does, then changing its coefficient
#should have some impact on the estimates as a whole. So I think it's running a placebo test 500 times.
# The p-test then compares the sum of all 500 permutation estimates (as absolute values) against the
#observed naive estimate ("real" outcome if the hypothesized relationship between the IV and DV is true).
#Since the p-value is significant, the null hypothesis (that there is NO treatment effect) can be rejected(?)
#The histogram also shows that the observed naive estimate of 1.45 lies way beyond the range of
#the permutated distribution. This occurs even with many instances of random assignment, which shows that the 
#1.45 coefficient cannot be attributed to random error or spuriousness. 
#Placebo tests are great for detecting any problems with data because the researcher is expecting
#that a placebo tests will yield no significant results whatsoever. If something does pop up, that's an indication
#that the data needs to be checked for any unusual outliers, any poorly measured concepts in the research design,
#a mistake in the randomization process, etc. 

