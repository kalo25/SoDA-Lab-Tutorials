###############################################################################
# Web Scraping + Google Scholar Tutorial: R
# Homework Assignment--Web Scrape for 10 Professors from your department
# Author: Jared Edgerton -- Kawain Lo
# Date: Sys.Date()
# See attached .txt file in github folder for answers to HW questions 1 and 4.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.

install.packages(c("rvest", "dplyr", "ggplot2", "scholar", "stringr"))
                 
install.packages(c("tibble"))

library(rvest)

library(dplyr)

library(ggplot2)
library(scholar)
library(stringr)
library(tibble)

install.packages(c("ggplot2"))
install.packages(c("vctrs"))
install.packages(c("cli"))
install.packages(c("rlang"))

library(ggplot2)
library(vctrs)
library(cli)
library(rlang)

# -----------------------------------------------------------------------------
#  Pulling Google Scholar Data for 10 Professors (Citations Over Time)
# -----------------------------------------------------------------------------
# Goal:
# - For each professor, we will:
#   (1) Define the Google Scholar ID
#   (2) Pull a profile summary
#   (3) Pull publications (and view the first 5)
#   (4) Pull citation history by year
#   (5) Combine all citation histories into one table and plot them

# -----------------------------------------------------------------------------
# Step 1: Hard-code Google Scholar IDs
# -----------------------------------------------------------------------------
staff_scholar_id <- "nm4ZRCgAAAAJ"
ramey_scholar_id <- "mhJ8jIwAAAAJ"
nguyen_scholar_id <- "PhLj9kIAAAAJ"
luo_scholar_id <- "J9NETMYAAAAJ"
graif_scholar_id <- "ijQOa5oAAAAJ"
gabbidon_scholar_id <- "tj2iWnQAAAAJ"
galvin_scholar_id <- "SKSJaMsAAAAJ"
damaske_scholar_id <- "zTqwiBYAAAAJ"
brothers_scholar_id <- "mFp3rBQAAAAJ"
wilcox_scholar_id <- "KVMRqDoAAAAJ"

# -----------------------------------------------------------------------------
# Step 2: Pull Google Scholar profiles (sequentially)
# -----------------------------------------------------------------------------
staff_profile <- get_profile(staff_scholar_id)
ramey_profile <- get_profile(ramey_scholar_id)
nguyen_profile <- get_profile(nguyen_scholar_id)
luo_profile <- get_profile(luo_scholar_id)
graif_profile <- get_profile(graif_scholar_id)
gabbidon_profile <- get_profile(gabbidon_scholar_id)
galvin_profile <- get_profile(galvin_scholar_id)
damaske_profile <- get_profile(damaske_scholar_id)
brothers_profile <- get_profile(brothers_scholar_id)
wilcox_profile <- get_profile(wilcox_scholar_id)

cat("\n------------------------------\n")
cat("Google Scholar Profile Summaries\n")
cat("------------------------------\n")

staff_name <- "Jeremy Staff"
ramey_name <- "Dave Ramey"
nguyen_name <- "Holly Nguyen"
luo_name <- "Liying Luo"
graif_name <- "Corina Graif"
gabbidon_name <- "Shaun Gabbidon"
galvin_name <- "Miranda Galvin"
damaske_name <- "Sarah Damaske"
brothers_name <- "Sarah Brothers"
wilcox_name <- "Pamela Wilcox"

cat("\n", staff_name, "\n", sep = "")
print(staff_profile)

cat("\n", ramey_name, "\n", sep = "")
print(ramey_profile)

cat("\n", nguyen_name, "\n", sep = "")
print(nguyen_profile)

cat("\n", luo_name, "\n", sep = "")
print(luo_profile)

cat("\n", graif_name, "\n", sep = "")
print(graif_profile)

cat("\n", gabbidon_name, "\n", sep = "")
print(gabbidon_profile)

cat("\n", galvin_name, "\n", sep = "")
print(galvin_profile)

cat("\n", damaske_name, "\n", sep = "")
print(damaske_profile)

cat("\n", brothers_name, "\n", sep = "")
print(brothers_profile)

cat("\n", wilcox_name, "\n", sep = "")
print(wilcox_profile)

# -----------------------------------------------------------------------------
# Step 3: Pull Google Scholar publications (sequentially)
# -----------------------------------------------------------------------------
staff_pubs <- get_publications(staff_scholar_id)
ramey_pubs <- get_publications(ramey_scholar_id)
nguyen_pubs <- get_publications(nguyen_scholar_id)
luo_pubs <- get_publications(luo_scholar_id)
graif_pubs <- get_publications(graif_scholar_id)
gabbidon_pubs <- get_publications(gabbidon_scholar_id)
galvin_pubs <- get_publications(galvin_scholar_id)
damaske_pubs <- get_publications(damaske_scholar_id)
brothers_pubs <- get_publications(brothers_scholar_id)
wilcox_pubs <- get_publications(wilcox_scholar_id)


cat("\n------------------------------\n")
cat("Recent Publications (first 5)\n")
cat("------------------------------\n")

cat("\n", staff_name, "\n", sep = "")
print(head(staff_pubs, 5))

cat("\n", ramey_name, "\n", sep = "")
print(head(ramey_pubs, 5))

cat("\n", nguyen_name, "\n", sep = "")
print(head(nguyen_pubs, 5))

cat("\n", luo_name, "\n", sep = "")
print(head(luo_pubs, 5))

cat("\n", graif_name, "\n", sep = "")
print(head(graif_pubs, 5))

cat("\n", gabbidon_name, "\n", sep = "")
print(head(gabbidon_pubs, 5))

cat("\n", galvin_name, "\n", sep = "")
print(head(galvin_pubs, 5))

cat("\n", damaske_name, "\n", sep = "")
print(head(damaske_pubs, 5))

cat("\n", brothers_name, "\n", sep = "")
print(head(brothers_pubs, 5))

cat("\n", wilcox_name, "\n", sep = "")
print(head(wilcox_pubs, 5))


# -----------------------------------------------------------------------------
# Step 4: Pull citation history (citations by year) and combine
# -----------------------------------------------------------------------------
staff_ct <- get_citation_history(staff_scholar_id) %>% mutate(name = staff_name)
ramey_ct <- get_citation_history(ramey_scholar_id) %>% mutate(name = ramey_name)
nguyen_ct <- get_citation_history(nguyen_scholar_id) %>% mutate(name = nguyen_name)
luo_ct <- get_citation_history(luo_scholar_id) %>% mutate(name = luo_name)
graif_ct <- get_citation_history(graif_scholar_id) %>% mutate(name = graif_name)
gabbidon_ct <- get_citation_history(gabbidon_scholar_id) %>% mutate(name = gabbidon_name)
galvin_ct <- get_citation_history(galvin_scholar_id) %>% mutate(name = galvin_name)
damaske_ct <- get_citation_history(damaske_scholar_id) %>% mutate(name = damaske_name)
brothers_ct <- get_citation_history(brothers_scholar_id) %>% mutate(name = brothers_name)
wilcox_ct <- get_citation_history(wilcox_scholar_id) %>% mutate(name = wilcox_name)


citation_df <- bind_rows(staff_ct, ramey_ct, nguyen_ct, luo_ct, graif_ct, gabbidon_ct, galvin_ct, damaske_ct, brothers_ct, wilcox_ct)

# Print the combined citation data
print(head(citation_df, 10))

# -----------------------------------------------------------------------------
# Step 5: Plot citations over time for each professor
# -----------------------------------------------------------------------------
ggplot(citation_df, aes(x = year, y = cites, color = name)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Google Scholar Citation History (Recent Years)",
    x = "Year",
    y = "Citations",
    color = "Soc-Crim Faculty"
  )

# -----------------------------------------------------------------------------
# Step 6: Median citations per year for each professor
# -----------------------------------------------------------------------------
median_cites <- citation_df %>%
  group_by(name) %>%
  summarize(median_cites = median(cites, na.rm = TRUE), .groups = "drop")

print(median_cites)

#------------------------------------------------------------------------------
#Research Interests of Professors #I tried this with Prof. Staff's page and couldn't figure out what was wrong with the HTML part. 
#------------------------------------------------------------------------------
staff_page <- read_html("https://sociology.la.psu.edu/people/jeremy-staff/") 

staff_text <- staff_page %>%
  html_element("body") %>%
  html_text(trim = TRUE)

staff_title <- str_extract(
  staff_text,
  "(Distinguished|Liberal Arts|Roy C\\.|Arnold S\\.|James P\\.)?\\s*(Associate\\s+)?Professor[^\\n\\r]{0,120}"
)

staff_title <- trimws(staff_title)

staff_email <- str_extract(staff_text, "[A-Za-z0-9._%+-]+@psu\\.edu")

staff_areas <- staff_page %>%
  html_elements(xpath = "//h2[normalize-space()='Research Interests']/following-sibling::ul[1]/li") %>%
  html_text(trim = TRUE)


staff_bio <- staff_page %>% #results show 2 rows of results
  html_elements(xpath = paste0(
    "//h2[normalize-space()='Professional Bio']/following-sibling::*[1]",
    " | //h2[normalize-space()='Research Interests']/following-sibling::*[1]"
  )) %>%
  html_text(trim = TRUE)

staff_interests <- paste(c(staff_areas), collapse = "; ")
staff_n_interest_items <- length(staff_areas)


staff_row <- tibble(
  scraped_interests = staff_interests,
  n_interest_items = staff_n_interest_items,
  bio = staff_bio 
)
print(staff_row)




scraped_profiles <- bind_rows(matt_row, sona_row, derek_row)

# Print the scraped data table
print(scraped_profiles)




