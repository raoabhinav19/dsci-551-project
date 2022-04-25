#!/usr/bin/env python
# coding: utf-8

# In[23]:


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import folium
import geopy
import ast
import warnings
from streamlit_folium import folium_static
from sklearn import preprocessing
import pickle 
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


# In[2]:


import boto
import boto.s3.connection
from io import StringIO
import boto3
import pandas as pd
import sys
from opencage.geocoder import OpenCageGeocode
from sklearn.metrics.pairwise import haversine_distances
from math import radians


# In[5]:


st.set_page_config(layout="wide")


# In[6]:


row1_1, row1_2 = st.columns((1, 1))


# In[7]:


with row1_1:
    st.title("SafePath Los Angeles")
    st.write(
        """
    ##
    Examining risk level of street roads in Los Angeles. 
    """
    )
    hour_selected = st.slider("Select hour of day", 0, 23)
    day_selected = st.selectbox("Pick a day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                           'Saturday', 'Sunday'])
    user_input = st.text_input("Enter street address here", 'Figueroa ST')


# In[8]:


keys = []


# In[9]:


with open('../data/api_keys.txt', 'r') as f:
    data = f.readlines()
    for i in data:
        keys.append(str(i).replace('\n', ''))


# In[10]:


conn = boto.connect_s3(
    aws_access_key_id = keys[0],
    aws_secret_access_key = keys[1]
)


# In[11]:


for bucket in conn.get_all_buckets():
    print("{name}\t{created}".format(
        name = bucket.name,
        created = bucket.creation_date,
    ))


# In[12]:


from io import StringIO


# In[13]:


client = boto3.client('s3', aws_access_key_id=keys[0], aws_secret_access_key=keys[1])
bucket_name = 'sofians3'
object_key = 'final_data.csv'
csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')


# In[61]:


class project_data:
    
    def __init__(self, user_input, day_selected, hour_selected):
        self.address = str(user_input).capitalize()
        self.day = day_selected
        self.hour = hour_selected
    def get_dataframe(self): 
        df_traffic = pd.read_csv(r'../data/dataProcessing_files/data_final.csv')       
        df_crash = pd.read_csv(r'../data/dataProcessing_files/kmeans_k=25.csv')
        return df_traffic, df_crash
        # function will get data for user input for appearing in pop-up text
    def get_risk(self, x):
        if x > 0 and x <= 10:
            return 'low'
        elif x > 11 and x <= 20:
            return 'medium'
        else:
            return 'high'
    def get_data(self):
        _, crash = self.get_dataframe()
        predicted_label = self.find_cluster()

        # Get mean traffic data by label
        dict_traffic = crash.groupby('label')['traffic_volume'].mean().to_dict()
        # Get mean accident data by label
        dict_accident = crash.groupby('label')['accident_count'].mean().to_dict()
        # Get risk level by label
        dict_risk = crash.groupby('label')['crash_rate'].mean().to_dict()
        risk = self.get_risk(dict_risk[predicted_label])
        return dict_traffic[predicted_label], dict_accident[predicted_label], risk

    # Get API
    def get_api(self):
        geocode_api = ''
        with open(r'../data/geocode_api_keys.txt', 'r') as f:
            geocode_api += str(f.read())
        
        return geocode_api

    # Geocoding...
    def coordinates(self):
        if ', Los Angeles' not in self.address:
            full_address = self.address + ', Los Angeles'
        key = self.get_api()
        geocoder = OpenCageGeocode(key)
        result = geocoder.geocode(full_address, no_annotations="1")
        if result and len(result):
            longitude = result[0]["geometry"]["lng"]  
            latitude = result[0]["geometry"]["lat"]
        
        else:
            return 'No location found'
        return [latitude, longitude]
    
    def open_model(self):
        return pickle.load(open("../results/kmeans.pkl", "rb"))
    

    # Finding the closest cluster and its relevant details
    def find_cluster(self):
        model = self.open_model()
        coordinates = self.coordinates()

        # Get the cluster
        prediction = model.predict(np.array([coordinates[0], coordinates[1]]).reshape(1, -1))[0]
        return prediction

    # Use this for getting data for plotting graph.
    def get_traffic_accident(self):
        data_,_ = self.get_dataframe()
        traffic_volume = pd.DataFrame(data_.groupby(['day_week', 'hour'])['mean_trafficVolume'].mean().unstack(level=-1).fillna(0))
        accident_volume = pd.DataFrame(data_.groupby(['day_week', 'hour'])['mean_accidentVolume'].mean().unstack(level=-1).fillna(0))
        return traffic_volume.loc[(self.day, self.hour)], accident_volume.loc[(self.day, self.hour)]
    
    # Use this for plotting stuff for all points on the maps
    # Do standard scaling for the data points, maybe save the weights for later
    def plot_maps(self):
        data_,_ = self.get_dataframe()
        scaler = StandardScaler()
        latitude = list(data_['accident_latitude'])
        longitude = list(data_['accident_longitude'])
        address = list(data_['accident_address'])
        day_week = list(data_['day_week'])
        traffic_vol = list(data_['mean_trafficVolume'])
        acc_count = list(data_['mean_accidentVolume'])
        hour = list(data_['hour'])
        crash_rate = scaler.fit_transform(np.array(data_['mean_crashRate']).reshape(-1,1))
        risk = data_['risk_level'].values.tolist()
        return [address, latitude, longitude, crash_rate, risk, day_week, traffic_vol, acc_count, hour]


# In[76]:


funcs = project_data(user_input, day_selected, hour_selected)
prediction_cluster = funcs.find_cluster()
location = funcs.coordinates()
color = "risk_level"
size = "acc_count"
popup = ["address","traffic_vol", "acc_count"]


# In[64]:


ans = funcs.plot_maps()
map_df = pd.DataFrame(ans).transpose()
map_df.columns = ['address', 'latitude', 'longitude', 'crash_rate', 'risk_level', 'day_week', 'traffic_vol', 'acc_count',  'hour']

# In[65]:

address = user_input + ', Los Angeles'
location_lat = list(map_df.loc[(map_df['address'] == address) & (map_df['day_week']==day_selected) & (map_df['hour']==hour_selected), 'latitude'])[0]
location_long = list(map_df.loc[(map_df['address'] == address) & (map_df['day_week']==day_selected) & (map_df['hour']==hour_selected), 'longitude'])[0]




copy = map_df.copy()


# In[66]:


lst_colors=["red","green","orange"]
lst_elements = sorted(list(map_df[color].unique()))
copy["color"] = copy[color].apply(lambda x: 
                lst_colors[lst_elements.index(x)])

scaler = preprocessing.MinMaxScaler()
copy["size"] = scaler.fit_transform(
               copy[size].values.reshape(-1,1)).reshape(-1)


# In[68]:


map_ = folium.Map(location=[location_lat, location_long], 
                      tiles='cartodbpositron', zoom_start=20)


# In[71]:


required_df = copy[(copy['day_week']==day_selected) & (copy['hour']==hour_selected)]


# In[77]:


required_df.apply(lambda row: folium.CircleMarker(
           location=[row["latitude"], row["longitude"]], popup=row[popup],
           color=row["color"], fill=True,
           radius=row["size"]).add_to(map_), axis=1)


# In[10]:

# In[82]:


with row1_2:
    
    st.subheader(
        f"""**Traffic Volume & Accident Count for {user_input} and {day_selected}**"""
    )

    chart_data = copy[(copy['address']==address)]
    chart_data= chart_data[(chart_data['day_week']==day_selected)][['hour', 'traffic_vol']]
    chart_data = pd.DataFrame(chart_data.groupby(by=['hour'])['traffic_vol'].sum(numeric_only=False))
    chart_data.reset_index(inplace=True)

    st.bar_chart(chart_data, width=500, height=500, use_container_width=True)


# In[80]:


row2_1, row2_2 = st.columns((1, 1))
with row2_1:
    st.write(
        f"""**{user_input} for {hour_selected}:00 and {day_selected}.**"""
    )
    folium_static(map_)


# In[81]:


with row2_2:
    chart_data_acc = copy[(copy['address']==address)]
    chart_data_acc = chart_data_acc[(chart_data_acc['day_week']==day_selected)][['hour', 'acc_count']]
    chart_data_acc = pd.DataFrame(chart_data_acc.groupby(by=['hour'])['acc_count'].sum(numeric_only=False))
    chart_data_acc.reset_index(inplace=True)
    
    st.bar_chart(chart_data_acc, width=500, height=500, use_container_width=True)

