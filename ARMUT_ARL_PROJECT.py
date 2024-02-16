import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# TASK 1: Data Preparation
#########################

# Step 1: Read the armut_data.csv file.
df_ = pd.read_csv("Armut ARL/armut_data.csv")
df = df_.copy()
df.head()

# Step 2: Create a new variable representing services by combining ServiceID and CategoryID with "_".
df["Service"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# Step 3: Create a new date variable containing only year and month information.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")

# Create a unique ID for each basket by combining UserID and the new date variable.
df["BasketID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

###############################################
# TASK 2: Generate Association Rules
###############################################

# Step 1: Create a pivot table of service baskets.

# Group by "BasketID" and "Service", count the number of services, and unstack to pivot the table.
invoice_product_df = df.groupby(['BasketID', 'Service'])['Service'].count().unstack().fillna(0)

# Convert the count values to binary (1 if the service is bought, 0 otherwise).
invoice_product_df = invoice_product_df.applymap(lambda x: 1 if x > 0 else 0)

invoice_product_df.head()

########################################################################
# Step 2: Generate association rules.
########################################################################

# Use the Apriori algorithm to find frequent itemsets.
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)

# Generate association rules.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

################################################################################################################################################
# Step 3: Provide service recommendations for a user who bought the service "2_0" within the last month.
################################################################################################################################################

# Define a function to recommend services.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

# Get service recommendations for the service "2_0".
arl_recommender(rules, "2_0", 3)
