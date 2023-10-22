# Databricks notebook source
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql import Window
import numpy as np
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

df = spark.read.option("header", True).option("inferSchema", True).csv('/mnt/edmentum-research/tmp/DataISU/Wavetronix_Nov_Dec 2023.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

print('Data overview')
df.printSchema()
print('Columns overview')
pd.DataFrame(df.dtypes, columns = ['Column Name','Data type'])

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

# config plots
sns.set(rc={'figure.figsize':(36, 26)})
sns.set_style("whitegrid")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero lane-occupancy

# COMMAND ----------

zero_occ = df.filter(f.col('lane-occupancy') == 0)

# COMMAND ----------

# check distributions
display(zero_occ.groupBy('lane-count').agg(f.count(f.lit('1'))))

# COMMAND ----------

display(zero_occ.groupBy('link-direction').agg(f.count(f.lit('1'))))

# COMMAND ----------

# check problematic devices
zero_occ_pd = zero_occ.filter(f.col('lane-count') != 0).groupBy('device-id').agg(f.count(f.lit('1')).alias('count')).toPandas()

# COMMAND ----------

zero_occ_pd.shape

# COMMAND ----------

zero_occ_pd = zero_occ_pd.sort_values(by=['count'], ascending=False)

# COMMAND ----------

zero_occ_pd.head(10)

# COMMAND ----------

ax = sns.barplot(data=zero_occ_pd, x="device-id", y="count")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero lane-count

# COMMAND ----------

zero_lc = df.filter(f.col('lane-count') == 0)

# COMMAND ----------

display(zero_lc.groupBy('link-direction').agg(f.avg('small-class-count').alias('avg_small_class_count'),
                                              f.avg('medium-class-count').alias('avg_medium_class_count'),
                                              f.avg('large-class-count').alias('avg_large_class_count')))

# COMMAND ----------


# # check problematic devices
display(zero_lc.filter((f.col('small-class-count') != 0) | (f.col('medium-class-count') != 0) | (f.col('large-class-count') != 0)).groupBy('device-id').agg(f.count(f.lit('1')).alias('count')).orderBy(f.col('count').desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date times

# COMMAND ----------

# explore
dt = df.select('date', 'time', 'start-ime', 'end-time', 'cst-time', 'month', 'day')

# COMMAND ----------

display(dt)

# COMMAND ----------

# correct col name
df = df.withColumnRenamed('start-ime', 'start-time').drop('start-ime')

# padding
df = df.withColumn("start_time_pad", f.lpad(f.col('start-time').cast('string'), 6, '0'))
df = df.withColumn("end_time_pad", f.lpad(f.col('end-time').cast('string'), 6, '0'))

# convert and extract time
df = df.withColumn("start_time_unix", f.from_unixtime(f.unix_timestamp(f.col("start_time_pad").cast('string'), "HHmmss")))
df = df.withColumn("end_time_unix", f.from_unixtime(f.unix_timestamp(f.col("end_time_pad").cast('string'), "HHmmss")))
df = df.withColumn('start_time_unix', f.split(df['start_time_unix'], ' ').getItem(1))
df = df.withColumn('end_time_unix', f.split(df['end_time_unix'], ' ').getItem(1))

# COMMAND ----------

display(df)

# COMMAND ----------

# drop cols
df = df.drop('start-time', 'end-time', 'start_time_pad', 'end_time_pad')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensor Reliability and Maintenance:
# MAGIC
# MAGIC - Can we identify sensors that require maintenance based on historical data, such as a declining working status?
# MAGIC - Are there patterns in sensor failures related to specific sensor types or locations?

# COMMAND ----------

# display distritbuions of categorial variables
display(df.groupBy('status').count())

# COMMAND ----------

sensor_fail_pd = df.filter(f.col('status')=='SENSOR_FAILURE').toPandas()

# COMMAND ----------

sensor_fail_pd['month_day'] = sensor_fail_pd['month'].astype('str') + sensor_fail_pd['day'].astype('str')
sensor_fail_pd['month_day'] = sensor_fail_pd['month_day'].astype('int')
sensor_fail_pd = sensor_fail_pd.sort_values(by='month_day', ascending=True)

# COMMAND ----------

sensor_fail_pd['month_day']

# COMMAND ----------

display(sensor_fail_pd.groupby(['month_day'])['month_day'].count().reset_index(name="count"))

# COMMAND ----------

sensor_fail_pd.groupby(['month_day'])['month_day'].count().reset_index(name="count").plot.bar(x='month_day', y='count', color='#FFAB00')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Traffic Pattern Analysis:
# MAGIC - How do traffic patterns (in terms of vehicle counts in a lane) change throughout the hour, day, and month?
# MAGIC - Are there noticeable trends in traffic based on the day of the week or month of the year?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time trend

# COMMAND ----------

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

# COMMAND ----------

# hourly sum
df_start_hour = df.withColumn('start_hour', f.split(f.col('start_time_unix'), ':').getItem(0))
df_start_hour = df_start_hour.withColumn('start_time_hour', f.concat(f.lit('2022'), f.lit('-'), 
                                              f.col('month').cast('string'), f.lit('-'), 
                                              f.col('day').cast('string'), f.lit(' '),
                                              f.col('start_hour').cast('string'), f.lit(':'),
                                              f.lit('00'), f.lit(':'),
                                              f.lit('00')))
df_start_hour = df_start_hour.withColumn('start_time_hour', f.to_timestamp(f.col('start_time_hour'), 'yyyy-MM-dd HH:mm:ss'))
df_start_hour_agg_pd = df_start_hour.groupBy('start_time_hour', 'detector-id').agg(f.sum('lane-count').alias('hourly-lane-count')).toPandas()

# COMMAND ----------

viz_df = df_start_hour_agg_pd.groupby(['start_time_hour']).agg({'hourly-lane-count': 'mean'}).reset_index().sort_values(by=['start_time_hour'])

# COMMAND ----------

viz_df

# COMMAND ----------

ax = sns.lineplot(data=viz_df, x="start_time_hour", y="hourly-lane-count")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

# COMMAND ----------

# daily sum
df = df.withColumn("day", f.lpad(f.col('day').cast('string'), 2, '0'))
df_start_day= df.withColumn('month_day', f.concat(f.lit('2022'), f.lit('-'), 
                                                  f.col('month').cast('string'), f.lit('-'), 
                                                  f.col('day').cast('string')))

# COMMAND ----------

df_start_day_agg = df_start_day.groupBy('month_day', 'detector-id').agg(f.sum('lane-count').alias('tot_lane_count'))
viz_df = df_start_day_agg.groupBy('month_day').agg(f.avg('tot_lane_count').alias('daily_lane_count'))
display(viz_df)

# COMMAND ----------

# break down by vehicle sizes
df_start_day_agg = df_start_day.groupBy('month_day', 'detector-id').agg(f.sum('small-class-count').alias('tot-small-class-count'),
                                                                          f.sum('medium-class-count').alias('tot-medium-class-count'),
                                                                          f.sum('large-class-count').alias('tot-large-class-count'))
viz_df = df_start_day_agg.groupBy('month_day').agg(f.avg('tot-small-class-count').alias('daily_small_class_count'),
                                                   f.avg('tot-medium-class-count').alias('daily_medium_class_count'),
                                                   f.avg('tot-large-class-count').alias('daily_large_class_count'))
display(viz_df)

# COMMAND ----------

df_start_day_agg = df_start_day.groupBy('month_day', 'detector-id').agg(f.sum('lane-occupancy').alias('tot_lane_occupancy'))
viz_df = df_start_day_agg.groupBy('month_day').agg(f.avg('tot_lane_occupancy').alias('daily_lane_occupancy'))
display(viz_df)

# COMMAND ----------

df_start_day_agg = df_start_day.groupBy('month_day', 'detector-id').agg(f.sum('lane-speed').alias('tot_lane_speed'))
viz_df = df_start_day_agg.groupBy('month_day').agg(f.avg('tot_lane_speed').alias('daily_lane_speed'))
display(viz_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bivariate relationship involving lane-speed
# MAGIC >TODO: identify and remove outliers

# COMMAND ----------

def find_outliers(df, cols):

    # Identifying the numerical columns in a spark dataframe
    numeric_columns = cols

    # Using the `for` loop to create new columns by identifying the outliers for each feature
    for column in numeric_columns:

        less_Q1 = 'less_Q1_{}'.format(column)
        more_Q3 = 'more_Q3_{}'.format(column)
        Q1 = 'Q1_{}'.format(column)
        Q3 = 'Q3_{}'.format(column)

        # Q1 : First Quartile ., Q3 : Third Quartile
        Q1 = df.approxQuantile(column,[0.25],relativeError=0)
        Q3 = df.approxQuantile(column,[0.75],relativeError=0)
        
        # IQR : Inter Quantile Range
        # We need to define the index [0], as Q1 & Q3 are a set of lists., to perform a mathematical operation
        # Q1 & Q3 are defined seperately so as to have a clear indication on First Quantile & 3rd Quantile
        IQR = Q3[0] - Q1[0]
        
        #selecting the data, with -1.5*IQR to + 1.5*IQR., where param = 1.5 default value
        less_Q1 =  Q1[0] - 1.5*IQR
        more_Q3 =  Q3[0] + 1.5*IQR
        
        isOutlierCol = 'is_outlier_{}'.format(column)
        
        df = df.withColumn(isOutlierCol,f.when((df[column] > more_Q3) | (df[column] < less_Q1), 1).otherwise(0))
    

    # Selecting the specific columns which we have added above, to check if there are any outliers
    selected_columns = [column for column in df.columns if column.startswith("is_outlier")]

    # Adding all the outlier columns into a new colum "total_outliers", to see the total number of outliers
    df = df.withColumn('total_outliers',sum(df[column] for column in selected_columns))

    # Dropping the extra columns created above, just to create nice dataframe., without extra columns
    df = df.drop(*[column for column in df.columns if column.startswith("is_outlier")])

    return df

# COMMAND ----------

# df_monthDay_outlier = find_outliers(df_monthDay, ['lane-speed', 'lane-count', 'lane-occupancy'])

# COMMAND ----------

# df_monthDay_outlier.count()

# COMMAND ----------

# df_monthDay_outlier = df_monthDay_outlier.filter(f.col('total_Outliers') == 0).drop('total_Outliers')

# COMMAND ----------

# df_monthDay_outlier.count()

# COMMAND ----------

df_start_day_agg = df_start_day.groupBy('detector-id', 'month_day').agg(f.sum(f.col('lane-speed')).alias('tot_lane_speed'),
                                                                                     f.sum(f.col('lane-count')).alias('tot_lane_count'),
                                                                                     f.sum(f.col('lane-occupancy')).alias('tot_lane_occupancy'))
df_start_day_agg_pd = df_start_day_agg.toPandas()

# COMMAND ----------

df_start_day_agg_pd['month_day'] = pd.to_datetime(df_start_day_agg_pd['month_day']).dt.date
df_start_day_agg_pd = df_start_day_agg_pd.sort_values(by=['month_day'])

# COMMAND ----------

ax = sns.boxplot(data=df_start_day_agg_pd, x="month_day", y="tot_lane_speed", color='#FFAB00')
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# COMMAND ----------

ax = sns.boxplot(data=df_start_day_agg_pd, x="month_day", y="tot_lane_count", color='#FFAB00')
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# COMMAND ----------

ax = sns.boxplot(data=df_start_day_agg_pd, x="month_day", y="tot_lane_occupancy", color='#FFAB00')
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# COMMAND ----------

sns.scatterplot(data=df_start_day_agg_pd, x="tot_lane_speed", y="tot_lane_count", color='#FFAB00')

# COMMAND ----------

sns.scatterplot(data=df_start_day_agg_pd, x="tot_lane_speed", y="tot_lane_occupancy", color='#FFAB00')

# COMMAND ----------

sns.scatterplot(data=df_start_day_agg_pd, x="tot_lane_count", y="tot_lane_occupancy", color='#FFAB00')

# COMMAND ----------

# MAGIC %md
# MAGIC ## choropleth (skipped)

# COMMAND ----------

# merge in location 
df = df.withColumn('device-id-0', f.split(df['device-id'], '-').getItem(0))
df = df.withColumn('device-id-1', f.split(df['device-id'], '-').getItem(1))
df = df.withColumn('device-id-2', f.split(df['device-id'], '-').getItem(2))
df = df.withColumn('device-id-merge', f.concat(f.col('device-id-0'), f.lit('-'), f.col('device-id-1')))

location = spark.read.option("header", True).option("inferSchema", True).csv('/mnt/edmentum-research/tmp/DataISU/wave_locations.csv')
merged = df.join(location, on = [df['device-id-merge'] == location['Infodevice-Id']], how='left')

# COMMAND ----------

display(merged)

# COMMAND ----------

merged_agg = merged.groupBy('infodevice-id', 'Latitude', 'Longitude').agg(f.sum('lane-count').alias('tot-lane-count'))
merged_agg_pd = merged_agg.toPandas()

# COMMAND ----------

merged_agg_pd

# COMMAND ----------

import plotly.express as px
import json

# Load the Iowa counties GeoJSON file
with open('/dbfs/mnt/edmentum-research/tmp/DataISU/County_Boundaries_of Iowa_20231021.geojson', 'r') as file:
    iowa_counties_geojson = json.load(file)

# # Create the choropleth map
# fig = px.choropleth(data_frame=merged_agg_pd,
#                     geojson=iowa_counties_geojson,
#                     locations=merged_agg_pd.index,
#                     featureidkey="properties.county_name",  # Specify the key in GeoJSON that matches county names
#                     color="tot-lane-count",
#                     hover_data=["Latitude", "Longitude", "tot-lane-count"],
#                     projection="mercator",
#                     title="Choropleth Map of Total Lane Counts in Iowa Counties")

# # Customize the map appearance
# fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")

# # Show the map
# fig.show()

# Create the choropleth map
fig = px.choropleth_mapbox(merged_agg_pd, geojson=iowa_counties_geojson, color="tot-lane-count",
                           color_continuous_scale="Viridis", range_color=(0, 100),
                           mapbox_style="carto-positron", zoom=6,
                           opacity=0.5, labels={"count": "tot-lane-count"})

# Update the layout and the geos
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_geos(fitbounds="locations", visible=False)

# Show the map
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine Learning

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString, VectorIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

class mlBuilder():
  """
  Currently, the class implements 3 spark ML (distributed) algorithms: linear regression, random forest, and gradient boosted tree
  df: spark data frame for buidling and evaluating ML models
  task: regression or classifiction (currently only supports regression)
  outcome: outcome variable to predict (e.g., avg_skill_score ,growth_score)
  ml_build(): provides trainig and test data from df for spark ML to train the models and prints evaluation metrics (RMSE, R2)
  get_feature_importance(): takes a model name as argument and outputs feature importance as dictionary
  plot_pred(): plot y vs. y_pred 
  """
  def __init__(self, df, task='regression', outcome='lane-occupancy'):
    self.df = df
    self.task = task
    self.outcome = outcome

  def ml_build(self):
    # drop na's
    df_nn = self.df.na.drop()

    # assemble the numeric cols into vectorized features
    self.numeric_cols = [field for (field, dataType) in df_nn.dtypes 
                    if ((dataType != "string") & (dataType != "timestamp") & (field != self.outcome))]
    vec_assembler = VectorAssembler(inputCols=self.numeric_cols , outputCol="features")

    self.featureDF = vec_assembler.transform(df_nn)

    # train test split
    (trainDF, testDF) = self.featureDF.randomSplit([.8, .2], seed=42)

    # perform regression
    if self.task == 'regression':
      # instantiate model
      rf = RandomForestRegressor(featuresCol="features", labelCol=self.outcome)
      lr = LinearRegression(featuresCol="features", labelCol=self.outcome)
      gbt = GBTRegressor(featuresCol="features", labelCol=self.outcome)

      # fit models
      self.rf_model = rf.fit(trainDF)
      self.lr_model = lr.fit(trainDF)
      self.gbt_model = gbt.fit(trainDF)

      # Make predictions.
      predictions_rf = self.rf_model.transform(testDF)
      predictions_lr = self.lr_model.transform(testDF)
      predictions_gbt = self.gbt_model.transform(testDF)

      # evaluate models
      evaluator_rmse = RegressionEvaluator(labelCol=self.outcome, predictionCol="prediction", metricName="rmse")
      evaluator_r2 = RegressionEvaluator(labelCol=self.outcome, predictionCol="prediction", metricName="r2")

      # display model performance
      rmse_rf = evaluator_rmse.evaluate(predictions_rf)
      rmse_lr = evaluator_rmse.evaluate(predictions_lr)
      rmse_gbt = evaluator_rmse.evaluate(predictions_gbt)
      print("Root Mean Squared Error (RMSE) on test data = %g from RF" % rmse_rf)
      print("Root Mean Squared Error (RMSE) on test data = %g from LR" % rmse_lr)
      print("Root Mean Squared Error (RMSE) on test data = %g from GBT" % rmse_gbt)

      r2_rf = evaluator_r2.evaluate(predictions_rf)
      r2_lr = evaluator_r2.evaluate(predictions_lr)
      r2_gbt = evaluator_r2.evaluate(predictions_gbt)
      print("R Squre (R2) on test data = %g from RF" % r2_rf)
      print("R Squre (R2) on test data = %g from LR" % r2_lr)
      print("R Squre (R2) on test data = %g from GBT" % r2_gbt)

      return [self.rf_model, self.lr_model, self.gbt_model]
    
  def get_feature_importance(self, model='RF'):
    # feature importance
    from itertools import chain

    attrs = sorted(
        (attr["idx"], attr["name"])
        for attr in (
            chain(*self.featureDF.schema["features"].metadata["ml_attr"]["attrs"].values())
        )
    ) 
    
    if model == 'RF':
      return [
          (name, self.rf_model.featureImportances[idx])
          for idx, name in attrs
          if self.rf_model.featureImportances[idx]
      ]
    if model == 'GBT':
      return [
          (name, self.gbt_model.featureImportances[idx])
          for idx, name in attrs
          if self.gbt_model.featureImportances[idx]
      ]
      
    if model == 'LR':
      return list(zip(self.numeric_cols, self.lr_model.coefficients))
    
    def plot_pred():
      pass

# COMMAND ----------

df_ml = df.select('cst-time', 'lane-id', 'device-id', 'detector-id', 'lane-speed', 'lane-occupancy')

# COMMAND ----------

ml = mlBuilder(df=df_ml, task='regression', outcome='lane-occupancy')

# COMMAND ----------

ml.ml_build()

# COMMAND ----------

ml.get_feature_importance(model='RF')

# COMMAND ----------

# MAGIC %md
# MAGIC >TODO: fit LSTM model

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as callbacks

from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM, Embedding
# from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

# COMMAND ----------



# COMMAND ----------

# Define your regression model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(cg_X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, activation='relu'))  # Use a linear activation for regression
model.add(Dense(1, activation='linear'))  # Linear activation for regression
model.summary()

# Compile the model for regression
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))  # Use mean squared error for regression

# Define callbacks
callbacks = [
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

# Train the regression model
history = model.fit(
    cg_X_train,
    cg_y_train,  # Replace with your target values for regression
    batch_size=64,
    epochs=10,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Traffic Speed and Congestion:
# MAGIC - Can we identify locations with persistent traffic congestion based on lane-speed data?
# MAGIC - Are there correlations between lane-speed and the number of vehicles in a lane?

# COMMAND ----------

# Sample code for traffic speed prediction using a decision tree regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Prepare the data and features for prediction
X = data[['lane-count', 'lane-occupancy', 'month', 'day', 'time']]
y = data['lane-speed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


# COMMAND ----------

# MAGIC %md
# MAGIC How can we predict the lane-speed based on the other features, such as lane-count, lane-occupancy, small-class-count, medium-class-count, and large-class-count? This is a regression problem that can be solved using various algorithms, such as linear regression, decision trees, random forests, or neural networks. One possible python code for this task is:

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("traffic_detector_data.csv")

# Select the features and the target
X = df[["lane-count", "lane-occupancy", "small-class-count", "medium-class-count", "large-class-count"]]
y = df["lane-speed"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual lane-speed")
plt.ylabel("Predicted lane-speed")
plt.title("Linear regression model")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Lane Occupancy Analysis:
# MAGIC - How does lane occupancy vary during different hours and days?
# MAGIC - Are there patterns indicating peak occupancy times in specific lanes?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vehicle Type Distribution:
# MAGIC
# MAGIC - What is the distribution of vehicle types (small, medium, large) on the roads at different times?
# MAGIC - Are there areas where large vehicles are more prevalent?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Temporal Traffic Predictions:
# MAGIC
# MAGIC - Can we build predictive models to forecast traffic conditions based on historical data, including time of day, month, and day of the week?
# MAGIC - How accurate are these predictions?
