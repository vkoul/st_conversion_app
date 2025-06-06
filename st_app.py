# Imports
import streamlit as st
import pandas as pd
import pickle

# setting the basic configuration of the web app. This is shown in the Tab
st.set_page_config(page_title = "Visitor Conversion Prediction" 
                    ,page_icon = ":bar_chart:")
                   # ,layout = "wide")

# page title 
st.title( "Will the website visitor convert? ✨")

with st.expander("About this app"):

    st.write("")

    st.markdown(
        """
This interactive app simulates a classification use case to predict whether a website visitor will convert based on key features.

Users can input the visitor's:

* Country of origin
* Traffic source (Ads, SEO, or Direct)
* User status (New or Returning)
* Age
* Number of pages visited

The prediction is made using a pre-trained classification model that identifies patterns in user behavior to estimate conversion likelihood. 
This is a practical demonstration of applying machine learning to real-world digital marketing and user analytics scenarios.
    """
    )



st.write("### Describe the visitor:")

# Creating columns to make the intake details look nicer
col1, col2, col3 = st.columns(3)

# First column: customer country?
with col1:
    customer_country = st.radio(
        label='Which Country?🌍', 
        options=['UK', 'US' ,'China' ,'Germany'])

# Second column: channel source
with col2:
    channel = st.radio(
        label='Which Source?🚀', 
        options = ['Ads', 'Seo', 'Direct'])

# Third column: new_user?
with col3:
    new_user = st.radio(
        label='New User?🙋‍♀️', 
        options = [1,0])  

# Age of the user
user_age = st.slider('How old is the visitor?🗓️', 3, 80, 2)

# how many pages visited?
pages_visited = st.slider('How many pages visited?📃', 1, 30, 1)


# # Cleaning up the details before putting into the df
# if customer_country == 'Germany':
#     country_Germany = 1
#     country_UK = 0
#     country_US = 0
    
# elif customer_country == 'UK':
#     country_Germany = 0
#     country_UK = 1

#     country_US = 0

# elif customer_country == 'China':
#     country_Germany = 0
#     country_UK = 0
#     country_US = 0

# else:
#     country_Germany = 0
#     country_UK = 0
#     country_US = 1


# # Source
# if channel == 'Seo':
#     source_Direct = 0
#     source_Seo = 1


# elif channel == 'Ads':
#     source_Direct = 0
#     source_Seo = 0

# else:
#     source_Direct = 1
#     source_Seo = 0

    

# # Creating the dataframe to run predictions on
# row = [user_age, new_user, pages_visited, country_Germany, country_UK, country_US, source_Direct, source_Seo]
# columns = [
#         'age', 'new_user', 'total_pages_visited', 'country_Germany', 
#         'country_UK', 'country_US', 'source_Direct', 'source_Seo']


# Mapping dictionaries for country and source
country_mapping = {
    'Germany': [1, 0, 0],
    'UK': [0, 1, 0],
    'China': [0, 0, 0],
    'US': [0, 0, 1]
}

source_mapping = {
    'Seo': [0, 1],
    'Ads': [0, 0],
    'Direct': [1, 0]
}

# Extract the country and source values from user inputs
country_values = country_mapping.get(customer_country, [0, 0, 0])
source_values = source_mapping.get(channel, [1, 0])

# Creating the dataframe to run predictions on
row = [user_age, new_user, pages_visited] + country_values + source_values
columns = ['age', 'new_user', 'total_pages_visited', 'country_Germany', 'country_UK', 'country_US', 'source_Direct', 'source_Seo']


visting_user = pd.DataFrame(dict(zip(columns, row)), index=[0])

# Show the table?
# st.table(visting_user)

# Now predicting!
if st.button(label="👆Click to Predict Conversion"):

    # Load the model
    loaded_model = pickle.load(open('rf_model_conversion.sav', 'rb'))
    
    # Make predictions (and get out pred probabilities)
    pred = loaded_model.predict(visting_user)[0]
    proba = loaded_model.predict_proba(visting_user)[:,1][0]
    
    # Sharing the predictions
    if pred == 0:
        st.write("### The visitor is not predicted to convert😔😭")
        st.write(f"Predicted probability of conversion: {proba*100:.2f} %")

    elif pred == 1:
        st.write("### The visitor is predicted to convert!🥳✨🎉")
        st.write(f"Predicted probability of conversion: {proba*100:.2f} %")
