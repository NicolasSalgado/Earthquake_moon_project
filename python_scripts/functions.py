import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

import sys 
import os

import matplotlib.pyplot as plt
import plotly.express as px

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

from scipy.fftpack import fft
from scipy.signal import blackman
from scipy.signal import periodogram

from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
from sklearn.preprocessing import MinMaxScaler

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\df")
from manage_file import FILES, getPath


random.seed(0)
def read_data(file="minable"):
    if file == "minable":
        minable = pd.read_csv(getPath(FILES.minable), index_col=0)
        return minable
    elif file == "earth":
        df = pd.read_excel(getPath(FILES.input_earthquake))
        df = df[["time", "year", "month", "day", "latitude", "longitude", "mag", "depth", "Pais"]]
        df.Pais = df.Pais.str.strip()
        df['time']= pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        df['time']= pd.to_datetime(df['time'])
        df.month = df.month.astype(int)
        df.day= df.day.astype(int)
        df = df.sort_values(by="time")
        df = df.reset_index()
        df = df[(df.year < 2023) & (df.year > 1972)]
        return df
    elif file == "moon":
        df_moon = pd.read_excel(getPath(FILES.input_moon))
        df_moon.columns = ["year", "day", "month", "acum_day", "ill_frac", "r/km", "dec", "ra/h", "ra/°"]
        df_moon.year = df_moon.year.replace({"common year": np.NaN, "leap year": np.NaN})
        df_moon.year.ffill(inplace=True)
        df_moon.year = df_moon.year.astype(int)
        df_moon.dropna(inplace=True)
        df_moon.month = df_moon.month.astype(int)
        df_moon.day= df_moon.day.astype(int)
        df_moon.acum_day= df_moon.acum_day.astype(int)
        return df_moon
    else:
        print(f"Error: {file} is not found. this function only work with 'minable', 'earth' and 'moon'" )

def magnitude_segmentation(df, mag_seg):

    df['MAG_SEG'] = [0]*len(df)
    for key, value in mag_seg.items():
        df.loc[(df['mag']<value[1]) & (df['mag']>=value[0]), 'MAG_SEG'] = key
    print(df["MAG_SEG"].value_counts())
    return df

def period_calculation(df, period_length=10):
    # according to the period length select we create the labels for each row in the dataset.
    start_year = df.time.iloc[0].year
    end_year = df.time.iloc[-1].year
    year_range = end_year - start_year
    modulo = year_range % period_length
    if modulo == 0:
        final_start = end_year - period_length
    else:
        final_start = end_year - modulo
    final_end = end_year+1
    if period_length == 1:
        starts = np.arange(start_year, final_start+1, period_length).tolist()
        tuples = [(start, start+period_length) for start in starts]
        # We'll add the last period calculated earlier
        tuples.append(tuple([final_start+1, final_end]))
    else:
        starts = np.arange(start_year, final_start, period_length).tolist()
        tuples = [(start, start+period_length) for start in starts]
        # We'll add the last period calculated earlier
        tuples.append(tuple([final_start, final_end]))
    bins = pd.IntervalIndex.from_tuples(tuples, closed='left')

    original_labels = list(bins.astype(str))
    new_labels = ['{} - {}'.format(b.strip('[)').split(', ')[0], int(b.strip('[)').split(', ')[1])-1) for b in original_labels]
    label_dict = dict(zip(original_labels, new_labels))
    # Assign each row to a period
    df['PERIOD'] = pd.cut(df['year'], bins=bins, include_lowest=True, precision=0)
    df['PERIOD'] = df['PERIOD'].astype("str")
    df = df.replace(label_dict)
    print(df.PERIOD.value_counts())
    return df

def histogram_monthly(df):
    df['NewDate'] = df['time'] + pd.offsets.DateOffset(days=-5)
    # extract the new month label from the shifted date
    df['NewMonth'] = df['NewDate'].dt.month
    # plot a histogram with Plotly
    fig = px.histogram(df.sort_values(by="NewMonth"), x="MAG_SEG", color='NewMonth', barmode='group', height=400)
    #fig.update_layout(xaxis_title='MAG_SEG', yaxis_title='Count')
    fig.show()

def histogram_countries(df, countries):
    df_ = df[df.Pais.isin(countries)]
    fig = px.histogram(df_, x="Pais",
                color='PERIOD', barmode='group',
                height=400)
    fig.show()
def histogram_cluster(df, clusters):
    df_ = df[df.cluster_label.isin(clusters)]
    df_.cluster_label = df_.cluster_label.astype(str) 
    fig = px.histogram(df_, x="cluster_label",
                color='PERIOD', barmode='group',
                height=400)
    fig.show()
def countries_value_counts(df):
    # To see number of earthquakes for each country.
    df.Pais = df.Pais.fillna("No_Country")
    for i,v in df.Pais.value_counts().items():
        print(f"{i} : {v}") 
    # Calculate the number of NAN in country column
    print(f'Number of NAN : {df.Pais.isna().sum()}')

def plot_map(df, clusters= False,specific_clusters=None):
    color_scale = [(0, 'orange'), (1,'red')]
    if not clusters:
        fig = px.scatter_mapbox(df, 
                                lat="latitude", 
                                lon="longitude", 
                                hover_name="index", 
                                hover_data=["index"],
                                zoom=1, 
                                height=800,
                                width=800)
    elif specific_clusters:
        df_filt = df[df.cluster_label.isin(specific_clusters)]
        fig = px.scatter_mapbox(df_filt, 
                                lat="latitude", 
                                lon="longitude", 
                                hover_name="index", 
                                hover_data=["index"],
                                color="cluster_label",
                                zoom=1, 
                                height=800,
                                width=800)
    else:
        fig = px.scatter_mapbox(df, 
                        lat="latitude", 
                        lon="longitude", 
                        hover_name="index", 
                        hover_data=["index"],
                        color="cluster_label",
                        zoom=1, 
                        height=800,
                        width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
def trendline(data, order=1):
    x_ = np.arange(0,len(data))
    coeffs = np.polyfit(x_, list(data), order)
    slope = coeffs[0]
    return float(slope)

def trendline_calculations(df, cluster = False):
    if cluster:
        neg_cluster ={}
        trend_results_cluster = {}
        for c in df["cluster_label"].unique().tolist():
            df_filt = df[df.cluster_label == c]
            series_totrend = df_filt.PERIOD.value_counts()
            series_totrend = series_totrend.sort_index()
            trend_results_cluster[c] = round(trendline(series_totrend),2)
        trend_results_cluster =dict(sorted(trend_results_cluster.items(), key=lambda item: item[1], reverse=True))
        for key, value in trend_results_cluster.items():
            print(f"For Cluster {key} the trendline is {value}")
            if value < 0:
                neg_cluster[key] = value
        return trend_results_cluster, neg_cluster
    else:
        series_totrend = df.PERIOD.value_counts()
        series_totrend = series_totrend.sort_index()
        print(f"Trendline for the whole series is : {round(trendline(series_totrend),2)}")
        trend_results = {}
        for mag_seg in df.MAG_SEG.unique().tolist():
            df_filt = df[df.MAG_SEG == mag_seg]
            series_totrend = df_filt.PERIOD.value_counts()
            series_totrend = series_totrend.sort_index()
            trend_results[f"Mag Segment {mag_seg}"] = round(trendline(series_totrend),2)
            
        for key, value in trend_results.items():
            print(f"For {key} the trendline is {value}")
        return trend_results

def calculate_clustering(df, normalized=False):
    df.dropna(axis=0,how='any',subset=['latitude','longitude'],inplace=True) 
    coords = df[['latitude', 'longitude']]
    if normalized:
        scaler = MinMaxScaler()
        coords = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters =60 , init ='k-means++')
    kmeans.fit(coords) # Compute k-means clustering.
    df['cluster_label'] = kmeans.fit_predict(coords)
    labels = kmeans.predict(coords) # Labels of each point
    centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
    return df

def best_number_of_clusters2(coords):
    K_clusters = range(10,70)
    kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    Y_axis = coords[['latitude']]
    X_axis = coords[['longitude']]
    score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
    # Visualize
    plt.plot(K_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()

def best_number_of_clusters1(coords):
    # Specify the range of k values to test
    k_range = range(10, 70)

    # Create a list to store the SSE values for each k
    sse = []

    # Loop over each value of k and compute the SSE
    for k in k_range:
        model = KMeans(n_clusters=k)
        model.fit(coords)
        sse.append(model.inertia_)

    # Plot the SSE values against k
    plt.plot(k_range, sse)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

def specific_cluster_info(df, cluster):
    df_filt = df[df.cluster_label == cluster]
    print(f"Cantidad de data del cluster {cluster} es {len(df_filt)}")
    series_totrend = df_filt.PERIOD.value_counts()
    series_totrend = series_totrend.sort_index()
    print(series_totrend)
    print(f"Cluster {cluster} has a trend of {trendline(series_totrend)}")

def interpolate_position(original_position, final_position, datetime):
    """
    Perform a linear interpolation of the position of the moon at a specific datetime.

    Parameters:
    original_position (float): The position of the moon at the beginning of the day.
    final_position (float): The position of the moon at the end of the day.
    datetime: The datetime in the format 'yyyy-mm-dd hh:mm:ss'.
    Returns:
    float: The interpolated position of the moon at the specified datetime.
    """

    # Extract year, day, and month information from the datetime object
    year, day, month = datetime.year, datetime.day, datetime.month


    # Compute the fraction of the day passed by
    time_passed = (datetime - datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    total_time = (pd.to_datetime(datetime.date()) + pd.DateOffset(days=1) - pd.to_datetime(datetime.date())).total_seconds()

    # Interpolate the position of the moon linearly based on the fraction of the day passed by
    interpolated_position = original_position + (final_position - original_position) * time_passed / total_time
    #print("datetime: ", datetime)
    #print(f"Original Position {original_position} - Final Position {final_position} - interpolated_position: {interpolated_position}")
    return interpolated_position

def apply_interpolation(df_merged, df_moon, vars_names=["r/km"]):
    for var_name in vars_names:
        df_merged[f"{var_name}_interpolated"] = [0]*len(df_merged)
    for index, row in df_merged.iterrows():
        next_day = pd.to_datetime(row["time"] + pd.DateOffset(days=1))
        try:
            next_row = df_moon[(df_moon.year == next_day.year) & (df_moon.month == next_day.month) & (df_moon.day == next_day.day)].iloc[0]
        except:
            next_row = None
        if (not np.isnan(row[var_name])) and (not next_row is None) :
            for var_name in vars_names:
                time= row["time"]
                original_pos = row[var_name]
                final_pos    = next_row[var_name]
                df_merged.loc[index,f"{var_name}_interpolated"] = interpolate_position(original_pos, final_pos, time)
            
    return df_merged