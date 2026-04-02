#######################################################
#Week 11 Homework: Databases and SQL 
######################################################
#------------------------------------------
#CONCEPTUAL QUESTION: Explain what a relational schema is and why it
#is useful for social data. In your answer, define primary keys and foreign keys, and 
#explain how they reduce duplication and enable joins. Use the (candidate, contributor, contribution)
#setting from this week's coding lab as your concrete example

#RESPONSE:-----------------------------------------------
#A relational schema is a set of design choices for the format and structure
#of a data table. These include the name of the table, the name of the columns/variables,
#the type of data each column is set to (text, integer, string, double, etc.), a primary key
#or unique identifier for each observation, a foreign key or a constructed variable/column that serves
#as a common reference point across multiple tables, and other specifications/qualifiers for the types
#of data and variables each data table can accept. All of these are necessary
#for big social data because of the sheer amount and complexity of datasets that researchers
#from this field work with. Imposing this kind of set schema upon all the different datasets you
#are analyzing saves you the trouble and hassle of remembering distinct styles/formats
#for each separate datatable. Primary and foreign keys in particular are important because the former
#represents each unique observation in a dataset, while the latter is an object shared between multiple
#datatables and linked to a specific variable (such as id/observation). The existence of a foreign
#key allows you to merge datatables together without worrying about duplicated observations because each observation's
#data can only be assigned to the specific foreign id associated with that table's unique id/observation identifier. 
#For example, in the class demo, the foreign key 'contributor_id' is directly linked
#to the 'id' variable in the table 'contributors', while the foreign key 'candidate_id' is directly
#linked to the 'id' variable in the table 'candidates'. These two keys are stored in the third table 'contributions',
#which serves as the reference point or base on which all other tables are merged into. 
#-------------------------------------------
#########################################################
#SETUP
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date, timedelta

#########################################################
#STEP 1: Build the database (campaign_finance.db) and load the synthetic tables titled
#candidates, contributors, and contributions
con = sqlite3.connect("hw_campaign_finance.db")
cur = con.cursor()


cur.execute("""                            
  CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    party TEXT, 
    office TEXT,
    winner INTEGER  -- 1 = winner, 0 = not winner (SQLite stores booleans as integers)
  );
""")

cur.execute("""
  CREATE TABLE contributors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    occupation TEXT,
    employer TEXT,
    state TEXT
  );
""")
#FOREIGN KEY--allows you to link your original table with other tables based on a common variable/column

cur.execute("""
  CREATE TABLE contributions (
    id INTEGER PRIMARY KEY,
    contributor_id INTEGER,
    candidate_id INTEGER,
    amount REAL,
    date TEXT,
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
  );
""")
con.commit()

#-----------Generating synthetic data

np.random.seed(123)

# ---- Candidates table (100 candidates) ----
candidate_ids = np.arange(1, 101)

candidate_names = np.array([f"Candidate {i}" for i in candidate_ids])

candidate_parties = np.random.choice(
    ["Democrat", "Republican", "Independent"],
    size=100,
    replace=True,
    p=[0.45, 0.45, 0.10]
)

candidate_offices = np.random.choice(
    ["Senate", "House", "Governor", "State Senate", "State House"],
    size=100,
    replace=True
)

candidate_winner = np.random.choice(
    [1, 0],
    size=100,
    replace=True,
    p=[0.5, 0.5]
)

candidates = pd.DataFrame({
    "id": candidate_ids,
    "name": candidate_names,
    "party": candidate_parties,
    "office": candidate_offices,
    "winner": candidate_winner
})

# ---- Contributors table (100,000 contributors) ----
contributor_ids = np.arange(1, 100001)

contributor_names = np.array([f"Contributor {i}" for i in contributor_ids])

contributor_occupations = np.random.choice(
    ["Engineer", "Teacher", "Doctor", "Lawyer", "Business Owner"],
    size=100000,
    replace=True
)

contributor_employers = np.array([f"Company {i}" for i in np.random.randint(1, 5001, size=100000)])

state_abb = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

contributor_states = np.random.choice(state_abb, size=100000, replace=True)

contributors = pd.DataFrame({
    "id": contributor_ids,
    "name": contributor_names,
    "occupation": contributor_occupations,
    "employer": contributor_employers,
    "state": contributor_states
})

# ---- Contributions table (1,000,000 contributions) ----
# - amount is log-normal to mimic a skewed donation distribution
# - date is sampled uniformly across 2024
contribution_ids = np.arange(1, 1000001)

contribution_contributor_ids = np.random.randint(1, 100001, size=1000000)
contribution_candidate_ids = np.random.randint(1, 101, size=1000000)

contribution_amounts = np.round(
    np.random.lognormal(mean=np.log(1000), sigma=1, size=1000000),
    2
)

start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
n_days = (end_date - start_date).days + 1

random_day_offsets = np.random.randint(0, n_days, size=1000000)
contribution_dates = np.array([(start_date + timedelta(days=int(d))).isoformat() for d in random_day_offsets])

contributions = pd.DataFrame({
    "id": contribution_ids,
    "contributor_id": contribution_contributor_ids,
    "candidate_id": contribution_candidate_ids,
    "amount": contribution_amounts,
    "date": contribution_dates
})
#----------------------------inserting synthetic data into database
candidates.to_sql("candidates", con, if_exists="append", index=False, chunksize=5000) 
contributors.to_sql("contributors", con, if_exists="append", index=False, chunksize=5000)
contributions.to_sql("contributions", con, if_exists="append", index=False, chunksize=5000)
con.commit()

#report the row counts in each table using SELECT COUNT(*)
print(pd.read_sql_query("SELECT COUNT(*) AS n_candidates FROM candidates;", con))
print(pd.read_sql_query("SELECT COUNT(*) AS n_contributors FROM contributors;", con))
print(pd.read_sql_query("SELECT COUNT(*) AS n_contributions FROM contributions;", con))

#show the schema for each table using PRAGMA table_info([table name])
print(pd.read_sql_query("PRAGMA table_info(candidates)", con))
print(pd.read_sql_query("PRAGMA table_info(contributors)", con))
print(pd.read_sql_query("PRAGMA table_info(contributions)", con))

#Explain in 2-4 sentences how contributor_id and candidate_id connect the tables
#--------------------------
# contributor_id and candidate_id are two columns/variables/objects generated based on the 'id' variable
#from each table, linked to each unique observation in a table. They are two objects that all
#tables have in common, and therefore can be linked to each other through these objects.
#For example, contributor_id for observation #1 (id value = 1) could be coded as '20', and this would be
#true across all three tables. 

##########################################################
#STEP 2: write a SQL query that uses at least one join and one aggregation:
    #REQUIRED: join 'contributions' to 'candidates' and compute
    #total contributions by party 
print(pd.read_sql_query("""
  SELECT
    co.id,
    co.candidate_id,
    ca.name AS candidate_name,
    co.amount,
    co.date
  FROM contributions co                      
  JOIN candidates ca                         
    ON co.candidate_id = ca.id               
  LIMIT 5;
""", con))

    #REQUIRED: restrict to 'contributions' with amount>1000

party = """
SELECT
  ca.party,
  SUM(co.amount) AS total_amount
FROM contributions co
JOIN candidates ca
  ON co.candidate_id = ca.id
WHERE co.amount > 1000
GROUP BY ca.party;
"""
total_amount = pd.read_sql_query(party, con)

#OUTPUT: a clean table with columns 'party', 'total_amount', and 'num_contributions'
party = """
SELECT
  ca.party,
  SUM(co.amount) AS total_amount,
  COUNT(*) AS num_contributions
FROM contributions co
JOIN candidates ca
  ON co.candidate_id = ca.id
WHERE co.amount > 1000
GROUP BY ca.party
ORDER BY total_amount DESC;
"""
total_amount = pd.read_sql_query(party, con)

print("\n------------------------------")
print("Total contribution amount by party")
print("------------------------------")
print(total_amount)

#Visualization: make a simple bar plot of 'total_amount' by party 
# Plot: total amount by party

plt.figure()
plt.bar(total_amount["party"], total_amount["total_amount"])
plt.title("Total Contributions by Party")
plt.xlabel("Party")
plt.ylabel("Total Amount ($)")
plt.tight_layout()
plt.savefig("11_databases_and_sql/problem_set/hw_contributions_by_party.png", dpi=150)
plt.show()

print("\nSaved plot: hw_contributions_by_party.png")

##########################################################
#STEP 3: Using SQL statements, do the following:
#-------------------------------------------------------
#Creating the index
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_contributor_id ON contributions (contributor_id);") 
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_candidate_id   ON contributions (candidate_id);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_amount         ON contributions (amount);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_contrib_date           ON contributions (date);")
con.commit()
#--------------------------------------------------------
    #verify which indexes exist on 'contributions' (query 'sqlite_master')
print(pd.read_sql_query("""
                        SELECT type, name, tbl_name, sql
                            FROM sqlite_master 
                            WHERE type = 'index';
                        """, con))


#choose one query that filters by candidate_id OR date OR amount. Run 'EXPLAIN QUERY PLAN'
print(pd.read_sql_query("EXPLAIN QUERY PLAN SELECT candidate_id FROM contributions", con))


#Explain in 4-6 sentences, interpret the query plan: does SQLite report using
#an index? if not, what index might help and why?
#--------------------------RESPONSE
#The command "EXPLAIN QUERY PLAN" shows you the actions that the SQL program
#would be taking if you ran the line of code that comes after "EXPLAIN QUERY PLAN".
#Because we created an index, the SQL report scans the data through the index (idx_contrib_contributor_id) instead
#of going through the entire original datatable. This is more efficient for conserving computing power.
