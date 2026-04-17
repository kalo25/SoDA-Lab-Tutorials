#########################################################
#Week 13 HW: Deterministic and Probabilistic Matching
#SoDA 501
#Kawain Lo
#########################################################
#CONCEPTUAL QUESTION
#Matching errors are often NOT RANDOM. Give two reasons why 
#record linkage error might vary across individuals or groups
#i.e., name commonness, transliteration, data-entry error, moving/ZIP changes).
#Explain one implication for social science inference if linkage 
#quality differs systematically across groups.
#---------------------------------------------------------------
#RESPONSE BELOW:
#-----------------
#One reason for record linkage error is differences in how various institutions collect and label data.
#For example, location identifiers can vary drastically between local-level and federal-level police datasets #Location
#variables can be practically anything, from zip codes, cross streets, exact addresses, city council districts, and police precincts to 
#entire metropolitan areas). In other cases, institutions switching from one system to another for data records also introduce the
#same problems. What makes this worse is that police data (and many other forms of administrative data) are notoriously opaque and disorganized. 

#The second reason for record linkage error may simply be variations in place or person names. For example, 
#names translated from foreign languages like Chinese into English lose important information--the meaning of each
#character, the tone, even the order of given vs family name. Two Chinese men named "Mark Zhang" who share the same birth
#year and live in the same zip code could be mistakenly identified as the same individual, just because they coincidentally
#have identical English first names and seemingly identical Chinese surnames.


#Linkage quality differing systematically across groups will directly affect how a researcher chooses to design
#their models (and maybe even formulate a research question). If a researcher wants to include a specific understudied population (such as a subgroup
#of undocumented immigrants), there's a high likelihood that any data on that group will be sparse. This may then
#discourage the researcher from including that population in the study at all, which then creates a feedback loop in which
#lack of scientific interest in x reduces the likelihood of high-quality, nationally-representative surveys including that
#very same group. Even if the researcher chooses to use the data, any conclusions drawn from those models will be much
#more limited in scope and depth because of the poor data quality--the researcher will be unable to concretely
#identify whether xyz variable exerts a distinct and unique effect on x individuals/groups.

##########################################################################################
#SETUP
#-----------------
library(tidyverse)
library(dplyr)       
library(ggplot2)
library(fastLink)
library(stringdist)
library(lubridate)
library(readr)

#########################################################################################
#EXERCISE 1:
#Generate or load the synthetic data + deterministic matching. Run the
#provided script to generate and save a dataset_a.csv and dataset_b. csv, then
#load them into R.
##########################################################################################
df_a <- read_csv("C:/Users/karra/Desktop/Coding_work/soda_501/13_record_linkage/dataset_a.csv")
df_b <- read_csv("C:/Users/karra/Desktop/Coding_work/soda_501/13_record_linkage/dataset_b.csv")

#perform deterministic matching on firstname, lastname, birthyear, and zipcode (merge)
df_merged <- df_a %>%
  inner_join(., df_b, by = c("firstname", "lastname", "birthyear", "zipcode"))


#Report the number of deterministic matches and the match rate (matches divided by
#nrow(df_a))

#the final merged dataframe has 6013 observations
match_rate <- ((nrow(df_merged))/(nrow(df_a)) * 100)
match_rate #60.13% match

#--------------------------------------------------------------------------------
#INTERPRETATION: In 3 to 6 sentences, explain why the deterministic match count looks the
#way it does in this simulation:
#-----------------------------
#The match rate is not as high as we would like it to be because of the strict constraints that were
#imposed upon the merge process. Specifying the variables means that only observations that are *exact* 
#matches of each other will be successfully identified. So a single typo in one word under any of those
#four columns could mean that an entire observation is dropped from the merged dataset.

##################################################################################################
#EXERCISE 2:
#Probabilistic matching with fastLink and threshold curve. Using "fastLink", match df_a and df_b
#on "firstname" "lastname" "birthyear" and "zipcode"
#Use fastlink(...return.all = TRUE)

prob_match <- fastLink(df_a, df_b,
  varnames = c("firstname", "lastname", "birthyear", "zipcode"),
  return.all = TRUE)

#Use getMatches(..., threshold.match = t) for a grid of thresholds from 
#0 to 1 (in increments of 0.01)

t <- seq(0, 1, by = 0.01)

results <- lapply(t, function(thresh) {
  matches <- getMatches(prob_match, dfA = df_a, dfB = df_b,
    threshold.match = thresh)
  
  data.frame(threshold = thresh,
    n_matches = nrow(matches))
})

match_results <- do.call(rbind, results) #see grid here


#Create a plot of [number of matches VS threshold]

ggplot(match_results, aes(x = threshold, y = n_matches)) + geom_line() + geom_point() +
  labs(title = "Threshold Curve", x = "Threshold", y = "# of Matches")
ggplot

#--------------------------------------------------------------------------------------------------------
#INTERPRETATION: In 4-6 sentences, describe how and why the curve changes as the threshold increases. 
#--------------------------------------------------------------------------------------------------------
#The shape of the curve resembles a step function, in which the success level for matches remains relatively constant 
#over a specific interval, and then abruptly drops as the quality threshold increases. I assume that this is because 
#many of the pairs have similar or identical match probability values. So once the threshold passes a certain point,
#that entire group of observations will be dropped all at once. 



#######################################################################################################
#EXERCISE 3:
#Match Quality, choosing a threshold, and interpreting posteriors.
#Using the probabilistic matches, evaluate match quality as the threshold changes and justify a final choice.
###################################################################################################################

#Create a low-threshold "candidate match" set (i.e., threshold.match = 0.000001) and then group matches
#by posterior bins (ex: 0.0-0.1, 0.1-0,.2, ......., 0.9-1.0)

#low threshold

match_low <- getMatches(prob_match, dfA = df_a, dfB = df_b,
  threshold.match = 0.000001)
names(match_low)


match_low$bin <- cut(match_low$posterior,
  breaks = seq(0, 1, by = 0.1), include.lowest = TRUE, right = TRUE)

str(match_low)
names(match_low)


grouped <- match_low %>%
             group_by(bin) %>%
             summarise(n_matches = n(),
             avg_posterior = mean(posterior, na.rm = TRUE))
  
#merging on zipcode b/c it has no noise
pairs <- merge(df_a, df_b, by = "zipcode", suffixes = c("_a", "_b"))


#For each posterior bin, compute:
#Levenshtein distance for first names (i.e. stringdist(..., method = "lv"0))
fn <- stringdist(pairs$firstname_a, pairs$firstname_b, method = "lv")

#Levenshtein distance for last names
ln <- stringdist(pairs$lastname_a, pairs$lastname_b, method = "lv")

#absolute difference in birth year 
by_diff <- (pairs$birthyear_a) - (pairs$birthyear_b)

#Make at least one plot that shows how these distances relate to posterior scores (boxplots of distance by posterior bin)

str(pairs$posterior)
match_low$bin <- cut(match_low$posterior,
                     breaks = seq(0, 1, by = 0.1),
                     include.lowest = TRUE)


ggplot(match_low, aes(x = bin, y = fn)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "First name distance VS posterior bin",
       x = "posterior bin",
       y = "Levenshtein distance")

ggplot(pairs, aes(x = bin, y = ln)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Last name distance VS posterior bin",
       x = "posterior bin",
       y = "Levenshtein distance")

ggplot(pairs, aes(x = bin, y = by_diff)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Birth year difference VS posterior bin",
       x = "Posterior bin",
       y = "Absolute difference")


#-----------------------------------------------------------------------------------------------------------------------------
#INTERPRETATION:
#Based on your diagnostics, choose a threshold you would use for this dataset and defend your choice
#Address how deterministic VS probabilistic matching differ in the number of matches found,
#how changing the threshold affects both the number of matches and match quality,
#discuss at least 2 limitations/biases of each approach
#the relationship between string distance and the posterior/threshold measure
#---------------------------------------------------------------------------------------------------------------------------
#I couldn't figure out a way to manipulate the data to create the plots. I think there's something about this package (fastLink)
#that makes it impossible to compare Levenshtein values against posterior bins. 


#limitations/biases of deterministic vs probabilistic matching:

#Deterministic matching is likely to yield fewer matches because of the strict requirements and lack of flexibility
#for typos, formatting issues in the text, etc. This means that observations which may actually be matches can be
#overlooked during the matching process if the raw data is littered with inconsistencies, thereby artificially 
#deflating the size of the final dataset. 
#Probabilistic matching addresses these issues by allowing patterns/matches to emerge from the data itself based on 
#string distance. However, it also comes with two limitations: first, that the threshold needs to be entirely determined
#by the researcher, and that even a small shift left or right can mean hundreds or thousands of potential pairs get dropped
#from the data; second, that probabilistic matching suffers from selection bias. Names tied to immigrant groups, racial/ethnic
#minority status, or low-income (low documentation) groups are consistently excluded from the final datasets because
#the probabilistic matching algorithms are unable to recognize actual matched pairs within this category. 

#The posterior score indicates the distance between strings--the closer the distance, the higher the score. This indicates
#a likely match between the two strings/vectors. Having a higher threshold measure also increases the quality of the final dataset,
#but this may also result in more potential matches being excluded because the algorithm deems them problematic for one reason or another.



