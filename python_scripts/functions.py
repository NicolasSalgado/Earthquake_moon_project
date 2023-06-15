import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

import sys 
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

from scipy.fftpack import fft
from scipy.signal import blackman
from scipy.signal import periodogram
import plotly.graph_objects as go



from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
from sklearn.preprocessing import MinMaxScaler

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\df")
from manage_file import FILES, getPath
import warnings
warnings.filterwarnings('ignore')

random.seed(0)
random.seed(0)
def read_data(file="minable"):
    """
    Function to read the data.
    Args:
        file: (str) name of the dataset to read, by default "minable". Other options "earth", "moon".

    Returns: (DataFrame) dataframe.

    """
    if file == "minable":
        minable = pd.read_csv(getPath(FILES.minable), index_col=0)
        minable['time']= pd.to_datetime(minable['time'])
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
    """
    Args:
        df: (DataFrame) dataframe to use
        mag_seg: Magnitudes groups to label each earthquake

    Returns: Dataframe with MAG_SEG column generated according to the mag_seg

    """
    df['MAG_SEG'] = [0]*len(df)
    for key, value in mag_seg.items():
        df.loc[(df['mag']<value[1]) & (df['mag']>=value[0]), 'MAG_SEG'] = key
    print(df["MAG_SEG"].value_counts())
    return df

def period_calculation(df, period_length=10):
    """
    Args:
        df: Dataframe with data
        period_length: Period in years to generate the grouping

    Returns: Dataframe with column PERIOD generated.

    """
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

def histogram_monthly(df, date_off_set=False, bool_mag_seg=True):
    """
    Create an histogram plot grouped by month
    Args:
        df: Dataframe
        date_off_set: Boolean(True,False), by default False
        bool_mag_seg: Boolean(True, False), by default True

    Returns: None

    """
    if date_off_set:
        print("Here we dateoffset -5 days")
        df['NewDate'] = df['time'] + pd.offsets.DateOffset(days=-5)
        # extract the new month label from the shifted date
        df['NewMonth'] = df['NewDate'].dt.month
        # plot a histogram with Plotly
        if bool_mag_seg:
            fig = px.histogram(df.sort_values(by="NewMonth"), x="MAG_SEG", color='NewMonth', barmode='group', height=400)
        else:
            fig = px.histogram(df.sort_values(by="NewMonth"), x="NewMonth", color='NewMonth', barmode='group', height=400)          
    else:
        if bool_mag_seg:
            fig = px.histogram(df.sort_values(by="month"), x="MAG_SEG", color='month', barmode='group', height=400)
        else:
            fig = px.histogram(df.sort_values(by="month"), x="month", color='month', barmode='group', height=400)


    fig.show()

def histogram_countries(df, countries):
    """
    Create an histogram plot grouped by the countries given
    Args:
        df: Dataframe
        countries: (list)[] countries to show in the histogram

    Returns: None

    """
    df_ = df[df.Pais.isin(countries)]
    fig = px.histogram(df_, x="Pais",
                color='PERIOD', barmode='group',
                height=400)
    fig.show()
def histogram_cluster(df, clusters):
    """
    Create an histogram plot filtered by clusters given
    Args:
        df: Dataframe
        clusters: (list) [] name of countries to filter

    Returns: None

    """
    df_ = df[df.cluster_label.isin(clusters)]
    df_.cluster_label = df_.cluster_label.astype(str) 
    fig = px.histogram(df_, x="cluster_label",
                color='PERIOD', barmode='group',
                height=400)
    fig.show()
def countries_value_counts(df):
    """
    Calculate number of earthquake by country
    Args:
        df: Dataframe

    Returns: None

    """
    # To see number of earthquakes for each country.
    df.Pais = df.Pais.fillna("No_Country")
    for i,v in df.Pais.value_counts().items():
        print(f"{i:18s}: {v}") 
    # Calculate the number of NAN in country column
    print(f'Number of NAN : {df.Pais.isna().sum()}')
def plot_map_animation(df, specific_clusters=None, animation_frame="year"):
    """
    Plot map  enable to frame by column (year by default) and specifics clusters.
    Args:
        df: Dataframe
        specific_clusters: (list ) By default None. List of clusters to filter
        animation_frame: column name to execute the frame

    Returns: None
    """
    if specific_clusters != None:
        df_filt = df[df.cluster_label.isin(specific_clusters)]
    else:
        df_filt = df
    a = df_filt["cluster_label"].value_counts()
    df_filt["cluster_count"] = df_filt["cluster_label"].apply(lambda x: a[x])
    df_filt["norm_mag"] = (df_filt["mag"]-df_filt["mag"].min())/(df_filt["mag"].max()-df_filt["mag"].min()) +0.1
    fig = px.scatter_mapbox(df_filt, 
                            lat="latitude", 
                            lon="longitude", 
                            hover_name="index",
                            animation_frame=animation_frame,
                            hover_data=["index", "cluster_count", "time", "mag", "ill_frac_interpolated", "depth"],
                            color="mag",
                            size="norm_mag",
                            zoom=0.6, 
                            height=600,
                            width=800)
    fig.update_layout(coloraxis_showscale=True)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
def plot_map(df, clusters= False,specific_clusters=None):
    """
    Plot map distinguish by cluster and specific clusters.
    Args:
        df: Dataframe
        clusters: (Boolean) By default False. Whether to color earthquakes by cluster or not.
        specific_clusters:(List) By default None. Clusters to filter.

    Returns: None
    """
    color_scale = [(0, 'orange'), (1,'red')]
    if not clusters:
        fig = px.scatter_mapbox(df, 
                                lat="latitude", 
                                lon="longitude", 
                                hover_name="index", 
                                hover_data=["index", "time", "mag", "ill_frac_interpolated", "depth"],
                                zoom=0.5, 
                                height=800,
                                width=800)
    elif specific_clusters:
        df_filt = df[df.cluster_label.isin(specific_clusters)]
        a = df_filt["cluster_label"].value_counts()
        df_filt["cluster_count"] = df_filt["cluster_label"].apply(lambda x: a[x])
        fig = px.scatter_mapbox(df_filt, 
                                lat="latitude", 
                                lon="longitude", 
                                hover_name="index", 
                                hover_data=["index", "cluster_count","time", "mag", "ill_frac_interpolated", "depth"],
                                color="cluster_label",
                                zoom=0.5, 
                                height=800,
                                width=800)
        fig.update_layout(coloraxis_showscale=False)
    else:
        a = df["cluster_label"].value_counts()
        df["cluster_count"] = df["cluster_label"].apply(lambda x: a[x])
        fig = px.scatter_mapbox(df, 
                        lat="latitude", 
                        lon="longitude", 
                        hover_name="index", 
                        hover_data=["index", "cluster_count", "time", "mag", "ill_frac_interpolated", "depth"],
                        color="cluster_label",
                        zoom=0.5, 
                        height=800,
                        width=800)
        fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
def trendline(data, order=1):
    """
    To calculate trendile of a series.
    Args:
        data: Series of data.
        order: Order of the trendline to calculate. By default 1.

    Returns: (float): Slope

    """
    x_ = np.arange(0,len(data))
    coeffs = np.polyfit(x_, list(data), order)
    slope = coeffs[0]
    return float(slope)

def trendline_calculations(df, cluster = False):
    """
    Trendline calcualtions for PERIOD label.
    Args:
        df: Dataframe
        cluster: (boolean). Whether to do the calculation by cluster or not.

    Returns:trendline results

    """
    if cluster:
        neg_cluster ={}
        trend_results_cluster = {}
        for c in df["cluster_label"].unique().tolist():
            df_filt = df[df.cluster_label == c]
            series_totrend = df_filt.PERIOD.value_counts()
            series_totrend = series_totrend.sort_index()
            sum_series = np.sum(series_totrend)
            trend = trendline(series_totrend)
            trend_results_cluster[c] = (round(trend,2), sum_series)
        trend_results_cluster =dict(sorted(trend_results_cluster.items(), key=lambda item: item[1][0], reverse=True))
        for key, value in trend_results_cluster.items():
            print(f"For Cluster {key} the trendline is                 : {value[0]}")
            print(f"Cluster {key}: Proportion of trend and #eartquakes : {round(value[0]/value[1], 2)}")

            if value[0] < 0:
                neg_cluster[key] = value
        return trend_results_cluster, neg_cluster
    else:
        series_totrend = df.PERIOD.value_counts()
        series_totrend = series_totrend.sort_index()
        sum_series = np.sum(series_totrend)
        trend = trendline(series_totrend)
        print(f"Trendline for the whole series is               : {round(trend,2)}")
        print(f"The proportion of trendline and #eartquakes are : {round(trend/sum_series, 2)}")
        trend_results = {}
        for mag_seg in df.MAG_SEG.unique().tolist():
            df_filt = df[df.MAG_SEG == mag_seg]
            series_totrend = df_filt.PERIOD.value_counts()
            series_totrend = series_totrend.sort_index()
            sum_series = np.sum(series_totrend)
            trend = trendline(series_totrend)
            trend_results[f"Mag Segment {mag_seg}"] = round(trend, 2)
            print(f"Trendline for Mag Segment {mag_seg} is          : {round(trend,2)}")
            print(f"The proportion of trendline and #eartquakes are : {round(trend/sum_series, 2)}")
            
        #for key, value in trend_results.items():
        #    print(f"For {key} the trendline is {value}")
        return trend_results

def calculate_clustering(df, normalized=False):
    """
    Function to calculate clustering of the dataframe based on latitude and longitude
    Args:
        df: Dataframe
        normalized: (Boolean). Whether to normalized data or not before the clustering.

    Returns:Dataframe with cluster_label as new column.

    """
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
    """
    Function to calculate based on Elbow curve plot the best number of clusters.
    Args:
        coords: Data

    Returns: None

    """
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
    """
    Gives information of trend and proportion of an specific cluster
    Args:
        df: Dataframe
        cluster: (int) Cluster label.

    Returns: None

    """
    df_filt = df[df.cluster_label == cluster]
    print(f"Cantidad de data del cluster {cluster} es {len(df_filt)}")
    series_totrend = df_filt.PERIOD.value_counts()
    series_totrend = series_totrend.sort_index()
    print(series_totrend)
    trend = round(trendline(series_totrend),2)
    print(f"Cluster {cluster:<d} has a trend of {trend}, proportion is {round(trend/len(df_filt),2)}")

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
    """
    Use interpoalte_position to interpolate the whole data of earthquakes according to the moon database.
    Args:
        df_merged: Dataframe with earthquake data
        df_moon:  Dataframe with moon data
        vars_names: (List): By default["r/km"]. List of variables to interpolate. Must be in moon database.

    Returns: dataframe with columns interpolated.

    """
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


def describe_columns(df, columns, step_quantile=0.25, clusters=[]):
    """
    Computes descriptive statistics of each column in a pandas DataFrame.
    
    Parameters:
        df (pandas.DataFrame): A pandas DataFrame object.
        columns (list): list of columns.
        step_quantile (float): The step between each quantile to compute. Default is 0.25.
        clusters (list): list of clusters of interest.
    
    Returns:
        (pandas.DataFrame): A DataFrame object containing the computed statistics for each column.
    """
    if clusters:
        df = df[df.cluster_label.isin(clusters)]
    
    df = df[columns]
    
    quantiles = np.arange(step_quantile, 1+step_quantile, step_quantile)
    stats = {
        'count': df.count(),
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
    }
    
    for q in quantiles:
        stats[f'{q*100:.0f}%'] = df.quantile(q)
        
    stats['max'] = df.max()
    stats_df = pd.DataFrame(stats)
    
    # Round values to 2 decimal places
    stats_df = stats_df.round(2)
    return stats_df.transpose()
def calculo_distribution(df, var = "ill_frac_interpolated", num_bins=10, specific_cluster=None):
    """
    Calculates the proportion of earthquakes given a range.
    Args:
        df: Dataframe
        var: (str) name of the column to use
        num_bins: Number of bins to create
        specific_cluster: None by default. Number of clsuter label to filter

    Returns: None

    """
    if specific_cluster != None:
        df = df[df.cluster_label == specific_cluster]
    
    max_ = df[var].max()
    min_ = df[var].min()
    bin_width = (max_ - min_) / num_bins

    # Define the edges of the bins
    bin_edges = [min_] + [min_ + i * bin_width for i in range(1, num_bins)] + [max_]
    print(var)
    len_df = len(df)
    for i in range(len(bin_edges)-1):
        df_filt = df[(df[var]>=bin_edges[i])&(df[var]<=bin_edges[i+1])]
        print(f"Rango ({bin_edges[i]:.3f},{bin_edges[i+1]:.3f}): cantidad total {len(df_filt)}, proporción sobre el total {round(100*len(df_filt)/len_df,3)}%")
def plot_calculo_distribution(df, var="ill_frac_interpolated", histnorm="percent", num_bins=10, specific_cluster=None):
    """
    Histogram plot to the proportion distribution calcualted for certain number of bins
    Args:
        df: Dataframe
        var: (str) name of the column to use
        histnorm: (str). By default "percent". How to calcualte y-axis.
        num_bins: (int). By default 10. Number of bins.
        specific_cluster: (int) By default None. Whether filter for specific cluster.

    Returns: None

    """
    if specific_cluster != None:
        df = df[df.cluster_label == specific_cluster]
    fig = px.histogram(df,var,histnorm= histnorm, cumulative=False)

    max_ = df[var].max()
    min_ = df[var].min()
    bin_width = (max_ - min_) / num_bins
    fig.update_traces(xbins=dict( # bins used for histogram
            start=min_,
            end=max_,
            size=bin_width
        )) 
    fig.show()
def plot_monthly(df, years=(1990,2010)):
    """
    Bar plot monthly over a period of years
    Args:
        df: Dataframe
        years: (tuple). (Year beginning, Year end)

    Returns:

    """
    df = df[(df.year>=years[0]) & (df.year<=years[1])]
    counts = df.groupby(['year', 'month']).size().reset_index(name='count')

    colors = [
        "#00539CFF",      # red
        "#EEA47FFF",    # orange
        "#F96167",    # yellow
        "#FCE77D",      # green
        "#CCF381",      # blue
        "#4831D4",    # purple
        "#E2D1F9",    # magenta
        "#317773",  # pink
        "#990011FF",     # gray
        "#FF69B4",     # black
        "#00FFFF",   # steel blue
        "#FCEDDA",     # saddle brown
    ]
    

    # Create a Plotly bar chart with the year and month on the x-axis and the count on the y-axis
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts['year'].astype(str) + '-' + counts['month'].astype(str).str.zfill(2),
                        y=counts['count'],
                        marker=dict(color=counts['month'], colorscale="sunset"),
                        ))
    fig.update_layout(title='Number of Occurrences by Month and Year',
                    xaxis_title='Year-Month',
                    yaxis_title='Count')
    fig.show()

def distribution_plot(df, df1=None, var="ill_frac_interpolated", specific_cluster=None):
    """
    Plot of kernel density estimate of a certain variable
    Args:
        df: Dataframe
        var: column to calculate. By default "ill_frac_interpolated".
        specific_cluster: (int) By default None. Whether to filter by a cluster.

    Returns: None

    """
    if specific_cluster != None:
        df = df[df["cluster_label"]==specific_cluster]
    
    ax = df[var].plot(kind='kde', figsize=(25,15))
    ax.axvline(df[var].min(), color='red', linestyle='--')
    ax.axvline(df[var].max(), color='red', linestyle='--')
    plt.title(f"Kernel Density Estimate of {var}",fontsize = 20)
    if isinstance(df1, pd.DataFrame):
        df1[var].plot(kind='kde', figsize=(25,15))
        plt.legend(["Earthquake_data", "Lower Bound", "Upper Bound", "Moon_data"], prop={'size': 15})
    plt.show()
    #fig = ff.create_distplot([df[var]],["var"])
    #fig.show()
def histogram_overtime(df,var,specific_cluster=None, animation_frame="year"):
    """
    Histogram plot with animation frame over a column. By default over year.
    Args:
        df: Dataframe
        var: name of the column to use
        specific_cluster: (int) By default None. Whether to filter by a cluster.
        animation_frame: (str) By default "year". Label of column to frame.

    Returns: None

    """
    if specific_cluster != None:
        if isinstance(specific_cluster,list):
            df = df[df.cluster_label.isin(specific_cluster)]
        else:
            df = df[df.cluster_label == specific_cluster]
        title_l = f"Histogram for cluster {specific_cluster} over time"
        
        fig = px.histogram(df,[var],histnorm="probability density", color="cluster_label", barmode="overlay",
                       cumulative=False, animation_frame=animation_frame, title=title_l, marginal="box") 
    else:
        title_l = "Histogram for the whole dataset over time"
        fig = px.histogram(df,[var],histnorm="probability density", barmode="overlay",
                       cumulative=False, animation_frame=animation_frame, title=title_l, marginal="box")
    #'group', 'overlay' or 'relative'
    fig.show()
    
def histogram_animation(df, var="ill_frac_interpolated", nbins=None , histnorm="percent",animation_frame="year", range_x=None, range_y=None):
    """
    Histogram plot over a var with animation frame.
    Args:
        df: Dataframe
        var: (str) By default "ill_frac_interpolated".  name of the column to use
        nbins: (int). By default None. Number of bins.
        histnorm: (str). By default "percent". How to calcualte y-axis.
        animation_frame: (str) By default "year". Label of column to frame.
        range_x: (tuple) By default None. If None use range by default, if you give a tuple use the tuple.
        range_y: (tuple) By default None. If None use range by default, if you give a tuple use the tuple.

    Returns: None

    """
    if "mag_round" in animation_frame:
        df["mag_round"] = df.mag.round(decimals=1)
        df = df.sort_values(by="mag_round")
    elif "mag" in animation_frame:
        df = df.sort_values(by="mag")
 
    fig = px.histogram(df,var,histnorm=histnorm,nbins=nbins ,cumulative=False, animation_frame=animation_frame, range_x=range_x, range_y=range_y)
    fig.show()
def filter_dataframe(df,var = "ill_frac_interpolated" ,var_range = (0,100), mag_range=(3,10)):
    """
    Filter the df according to a var y magnitude over a certain range.
    Args:
        df: Dataframe.
        var: (str) By default "ill_frac_interpolated".  name of the column to use
        var_range: (tuple) By default (0,100). Range to filter the var.
        mag_range: (tuple) By default (3,10). Range to filter the magnitude var.

    Returns: Filtered Dataframe.

    """
    df_filt = df[(df[var] <var_range[1])&(df[var] >var_range[0])&(df.mag <mag_range[1])&(df.mag > mag_range[0])]
    print(f"Dataframe filtrado queda con {len(df_filt)} filas")
    return df_filt


def count_number_days(var="r/km", range_var=[]):
    """
    Calculate the number of days that a certaing variable (moon data) is in a range.
    Args:
        var: (str) By default "r/km". Column to use.
        range_var: list. Range to use.

    Returns:None

    """
    df_moon = read_data(file="moon")
    if len(range_var)>0:
        cant_filt = len(df_moon[(df_moon[var]<range_var[1])&(df_moon[var]>range_var[0])])
        cant_total = len(df_moon)
        print(f"Cantidad de días con {var} entre {range_var[0]} - {range_var[1]} es {cant_filt}")
        print(f"Porcentaje del total {round(100*cant_filt/cant_total,2)}%")

def filter_by_date(df, date_filt="2020-05-03", nweeks=4):
    """
    Filter dataframe over a Initial date and a number of weeks ahead
    Args:
        df: Dataframe
        date_filt: (str) By default "2020-05-03". Format "YYYY-MM-DD".
        nweeks: (int) By default 4.Number of weeks.

    Returns: filtered dataframe

    """
    time_filter = datetime.strptime(date_filt, "%Y-%m-%d")
    df_filt = df[df.time >= time_filter]
    delta = timedelta(days=nweeks*7)
    time_final = delta + time_filter
    def calcular_semana(fecha):
        semanas_pasadas = (fecha - time_filter).days // 7
        return min(semanas_pasadas + 1, nweeks)
    df_filt = df_filt[df_filt['time'] <= time_final]
    df_filt['week'] = df_filt['time'].apply(calcular_semana)
    return df_filt