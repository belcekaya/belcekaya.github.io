---
layout: archive-dates
title: PySpark Applications
toc: true
---

## Project Description
The tripdata_2017-01 dataset contains the yellow and green taxi trip records include fields capturing pick-up and drop-off dates/times, pick-up and drop-off locations, trip distances, itemized fares, rate types, payment types, and driver-reported passenger counts. The data used in the attached datasets were collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). 
- The dataset link: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

In described dataset, many data manipulation will be made to get useful information (such as Average speed of taxis in terms of the hour, momst common taxi trips, etc.) by using pyspark. Moreover, for each action, the execution time, amount of data processed and processing speed will be computed.


## Preparing the Spark configurations

First, Spark will be configured.



```python
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://dlcdn.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
!tar xf spark-3.1.2-bin-hadoop3.2.tgz
!pip install -q findspark
```


```python
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.1.2-bin-hadoop3.2"
```


```python
import findspark
findspark.init()
```


```python
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import *
import time

conf = SparkConf().set('spark.ui.port', '4050')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.master('local[*]').getOrCreate()

spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://2290828e1bdf:4050">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.1.2</code></dd>
      <dt>Master</dt>
        <dd><code>local[*]</code></dd>
      <dt>AppName</dt>
        <dd><code>pyspark-shell</code></dd>
    </dl>
</div>

    </div>




## Reading the csv file, analyzing the RDD.


```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").\
master("local[*]").getOrCreate()

df = spark.read.format("csv").\
load("tripdata_2017-01.csv",inferSchema =True,sep=",",header="true") #read the csv file.
print(df)

```

    DataFrame[VendorID: int, tpep_pickup_datetime: string, tpep_dropoff_datetime: string, passenger_count: int, trip_distance: double, RatecodeID: int, store_and_fwd_flag: string, PULocationID: int, DOLocationID: int, payment_type: int, fare_amount: double, extra: double, mta_tax: double, tip_amount: double, tolls_amount: double, improvement_surcharge: double, total_amount: double]
    


```python
df.dtypes
```




    [('VendorID', 'int'),
     ('tpep_pickup_datetime', 'string'),
     ('tpep_dropoff_datetime', 'string'),
     ('passenger_count', 'int'),
     ('trip_distance', 'double'),
     ('RatecodeID', 'int'),
     ('store_and_fwd_flag', 'string'),
     ('PULocationID', 'int'),
     ('DOLocationID', 'int'),
     ('payment_type', 'int'),
     ('fare_amount', 'double'),
     ('extra', 'double'),
     ('mta_tax', 'double'),
     ('tip_amount', 'double'),
     ('tolls_amount', 'double'),
     ('improvement_surcharge', 'double'),
     ('total_amount', 'double')]




```python
df.head()
```




    Row(VendorID=1, tpep_pickup_datetime='2017-01-09 11:13:28', tpep_dropoff_datetime='2017-01-09 11:25:45', passenger_count=1, trip_distance=3.3, RatecodeID=1, store_and_fwd_flag='N', PULocationID=263, DOLocationID=161, payment_type=1, fare_amount=12.5, extra=0.0, mta_tax=0.5, tip_amount=2.0, tolls_amount=0.0, improvement_surcharge=0.3, total_amount=15.3)



## **Average speed of taxis in terms of the hour.**

According to the observation to the dataset, to calculate the travel time, the drop-off_datetime and the pick-up_datetime columns are minused , but most of the cases the time unit is 'minute'. Therefore to uinform the time units first to seconds, the **unix_timestamp** function is used. Hence this step, the pick-up_datetime and drop-off_datetime into seconds are returned.


```python
df2 = df.select(
   col("VendorID").alias("ID"),  
   col("trip_distance").alias("distance"),
   hour(col("tpep_pickup_datetime")).alias("pickup_hour"),
   unix_timestamp(col("tpep_pickup_datetime")).alias("pickup_timestamp"),
   unix_timestamp(col("tpep_dropoff_datetime")).alias("dropoff_timestamp")
   )
start_time = time.time()
data_amount = df2.count()
df2.printSchema()
df2.show()
end_time = time.time()
print('The amount of data is: {}'.format(data_amount)) # amount of data processed in terms of the size of the data frame created
print('The execution time is: {}'.format(end_time-start_time)) # total time taken to perform the action
print('Speed: {}'.format(((end_time-start_time)/data_amount)))
```

    root
     |-- ID: integer (nullable = true)
     |-- distance: double (nullable = true)
     |-- pickup_hour: integer (nullable = true)
     |-- pickup_timestamp: long (nullable = true)
     |-- dropoff_timestamp: long (nullable = true)
    
    +---+--------+-----------+----------------+-----------------+
    | ID|distance|pickup_hour|pickup_timestamp|dropoff_timestamp|
    +---+--------+-----------+----------------+-----------------+
    |  1|     3.3|         11|      1483960408|       1483961145|
    |  1|     0.9|         11|      1483961547|       1483961761|
    |  1|     1.1|         11|      1483961900|       1483962125|
    |  1|     1.1|         11|      1483962733|       1483963056|
    |  2|    0.02|          0|      1483228800|       1483228800|
    |  1|     0.5|          0|      1483228802|       1483229030|
    |  2|    7.75|          0|      1483228802|       1483231162|
    |  1|     0.8|          0|      1483228803|       1483229218|
    |  1|     0.9|          0|      1483228805|       1483229313|
    |  2|    1.76|          0|      1483228805|       1483229104|
    |  2|    8.47|          0|      1483228805|       1483229736|
    |  1|     2.4|          0|      1483228806|       1483229516|
    |  1|    12.6|          0|      1483228806|       1483230217|
    |  1|     0.9|          0|      1483228806|       1483229333|
    |  2|    2.43|          0|      1483228806|       1483229370|
    |  2|     2.6|          0|      1483228806|       1483229765|
    |  2|    4.25|          0|      1483228806|       1483229892|
    |  2|    0.65|          0|      1483228807|       1483229262|
    |  2|    3.42|          0|      1483228809|       1483230861|
    |  1|     6.6|          0|      1483228810|       1483230292|
    +---+--------+-----------+----------------+-----------------+
    only showing top 20 rows
    
    The amount of data is: 971010
    The execution time is: 0.7318475246429443
    Speed: 7.536972066641377e-07
    

In this step, a new column 'speed_of_taxis_hr' is created by writing the basic formula which is 'distance/time difference' to calculate the average speed, among which, the time diffence=(dropoff_timestamp-pickup_timestamp)/3600, making the time units into hour.




```python
df3 = df2.withColumn('speed_of_taxis_hr', (df2['distance'] / ((df2['dropoff_timestamp'] - df2['pickup_timestamp'] )/3600)))
df3.filter("speed_of_taxis_hr is not Null").show()
```

    +---+--------+-----------+----------------+-----------------+------------------+
    | ID|distance|pickup_hour|pickup_timestamp|dropoff_timestamp| speed_of_taxis_hr|
    +---+--------+-----------+----------------+-----------------+------------------+
    |  1|     3.3|         11|      1483960408|       1483961145|16.119402985074625|
    |  1|     0.9|         11|      1483961547|       1483961761| 15.14018691588785|
    |  1|     1.1|         11|      1483961900|       1483962125|              17.6|
    |  1|     1.1|         11|      1483962733|       1483963056|12.260061919504645|
    |  1|     0.5|          0|      1483228802|       1483229030| 7.894736842105263|
    |  2|    7.75|          0|      1483228802|       1483231162|11.822033898305085|
    |  1|     0.8|          0|      1483228803|       1483229218|6.9397590361445785|
    |  1|     0.9|          0|      1483228805|       1483229313| 6.377952755905512|
    |  2|    1.76|          0|      1483228805|       1483229104|21.190635451505017|
    |  2|    8.47|          0|      1483228805|       1483229736| 32.75187969924812|
    |  1|     2.4|          0|      1483228806|       1483229516|12.169014084507042|
    |  1|    12.6|          0|      1483228806|       1483230217|32.147413182140326|
    |  1|     0.9|          0|      1483228806|       1483229333| 6.148007590132827|
    |  2|    2.43|          0|      1483228806|       1483229370| 15.51063829787234|
    |  2|     2.6|          0|      1483228806|       1483229765| 9.760166840458812|
    |  2|    4.25|          0|      1483228806|       1483229892|14.088397790055247|
    |  2|    0.65|          0|      1483228807|       1483229262| 5.142857142857143|
    |  2|    3.42|          0|      1483228809|       1483230861|               6.0|
    |  1|     6.6|          0|      1483228810|       1483230292| 16.03238866396761|
    |  1|     0.5|          0|      1483228810|       1483228969|11.320754716981131|
    +---+--------+-----------+----------------+-----------------+------------------+
    only showing top 20 rows
    
    

After that, df3 is grouped by the column 'pickup_hour' and to find the mean of speed of taxis, aggregate function is used. In the end, all RDD is sorted by pickup_hour in ascending order.


```python
df_avg_speed = df3.groupBy('pickup_hour').agg(mean('speed_of_taxis_hr').alias("Average Speed")).sort(asc("pickup_hour"))
df_avg_speed.filter("pickup_hour is not Null")
start_time=time.time()
data_amount = df_avg_speed.select(col("pickup_hour")).count()
df_avg_speed.show(24)
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
print('Number of pickup hours: {}'.format(data_amount))
print('The speed per hour group is {}'.format((end_time-start_time)/data_amount))
```

    +-----------+------------------+
    |pickup_hour|     Average Speed|
    +-----------+------------------+
    |          0|15.772703466103813|
    |          1|14.893825153701224|
    |          2|15.756147809843977|
    |          3|17.525558224313034|
    |          4| 20.43110338584045|
    |          5| 22.73870032615851|
    |          6| 19.46059842900451|
    |          7|14.693576421158458|
    |          8|14.160717306537206|
    |          9|13.465957878250988|
    |         10|14.408612028666614|
    |         11|13.260485301127076|
    |         12| 13.20056144899561|
    |         13|12.856997916279632|
    |         14|  14.2364434812878|
    |         15| 12.73385100734577|
    |         16|16.185655968202333|
    |         17|13.222232845495745|
    |         18|13.143430909065156|
    |         19|15.388002986253456|
    |         20|14.675625700100827|
    |         21|15.237883857610976|
    |         22| 17.18543859247738|
    |         23|19.342113607903748|
    +-----------+------------------+
    
    The execution time is: 9.170201063156128
    Number of pickup hours: 24
    The speed per hour group is 0.3820917109648387
    

Another way to do this equivelantly via a SQL query is the following:


```python
# create a new temp view
df.createOrReplaceTempView("taxis")
var1 = spark.sql("select  hour(to_timestamp(tpep_pickup_datetime)) as hour, trip_distance / ((unix_timestamp(to_timestamp(tpep_dropoff_datetime)) - unix_timestamp(to_timestamp(tpep_pickup_datetime))) / 3600) as speed from taxis")
var1.printSchema()
var1.show()
var1.createOrReplaceTempView("speedtable")
```

    root
     |-- hour: integer (nullable = true)
     |-- speed: double (nullable = true)
    
    +----+------------------+
    |hour|             speed|
    +----+------------------+
    |  11|16.119402985074625|
    |  11| 15.14018691588785|
    |  11|              17.6|
    |  11|12.260061919504645|
    |   0|              null|
    |   0| 7.894736842105263|
    |   0|11.822033898305085|
    |   0|6.9397590361445785|
    |   0| 6.377952755905512|
    |   0|21.190635451505017|
    |   0| 32.75187969924812|
    |   0|12.169014084507042|
    |   0|32.147413182140326|
    |   0| 6.148007590132827|
    |   0| 15.51063829787234|
    |   0| 9.760166840458812|
    |   0|14.088397790055247|
    |   0| 5.142857142857143|
    |   0|               6.0|
    |   0| 16.03238866396761|
    +----+------------------+
    only showing top 20 rows
    
    


```python
# get the average speed and display it in terms of the hour
var2 = spark.sql('select hour, avg(speed) from speedtable group by hour order by hour')
import time
start = time.time()
var2.show(24)
end = time.time()
duration = end - start
speed = duration / var2.count() 
print(f"Program execution time: {duration} speed: {speed} taxi drives per second")
```

    +----+------------------+
    |hour|        avg(speed)|
    +----+------------------+
    |   0|15.772703466103813|
    |   1|14.893825153701224|
    |   2|15.756147809843977|
    |   3|17.525558224313034|
    |   4| 20.43110338584045|
    |   5| 22.73870032615851|
    |   6| 19.46059842900451|
    |   7|14.693576421158458|
    |   8|14.160717306537206|
    |   9|13.465957878250988|
    |  10|14.408612028666614|
    |  11|13.260485301127076|
    |  12| 13.20056144899561|
    |  13|12.856997916279632|
    |  14|  14.2364434812878|
    |  15| 12.73385100734577|
    |  16|16.185655968202333|
    |  17|13.222232845495745|
    |  18|13.143430909065156|
    |  19|15.388002986253456|
    |  20|14.675625700100827|
    |  21|15.237883857610976|
    |  22| 17.18543859247738|
    |  23|19.342113607903748|
    +----+------------------+
    
    Program execution time: 5.12075662612915 speed: 0.21336485942204794 taxi drives per second
    

## **Most common taxi trips**

To find the most common taxi trips, the idea is to find the Drop location ID which appears the most times in the column. So first 'select' function is used to query the objective column. And later on the column "DOLocationID" renamed to "Drop_Location".


```python
#Most common taxi trips
import time
df_location = df.select( 
            col("DOLocationID").alias("Drop_Location")
)
start_time=time.time()
df_location.show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))

```

    +-------------+
    |Drop_Location|
    +-------------+
    |          161|
    |          234|
    |          161|
    |           75|
    |          234|
    |           48|
    |           36|
    |          161|
    |           50|
    |           74|
    |          262|
    |          236|
    |          265|
    |          186|
    |          107|
    |          163|
    |           36|
    |           68|
    |          148|
    |          232|
    +-------------+
    only showing top 20 rows
    
    The execution time is: 0.11707305908203125
    


```python
df5 = df_location.groupBy('Drop_Location').agg(count('Drop_Location').alias('Num_Trip')).sort(desc('Num_Trip'))
start_time = time.time()
data_amount = df5.count()
df5.show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
print('Number of pickup hours: {}'.format(data_amount))
print('The speed per hour group is {}'.format((end_time-start_time)/data_amount))
```

    +-------------+--------+
    |Drop_Location|Num_Trip|
    +-------------+--------+
    |          230|   31036|
    |           79|   30728|
    |           48|   29379|
    |          170|   28972|
    |          161|   28170|
    |          236|   27701|
    |          186|   27150|
    |          162|   25892|
    |          234|   24869|
    |          237|   24680|
    |          142|   23392|
    |           68|   22753|
    |          239|   22528|
    |          141|   22089|
    |          107|   21997|
    |          246|   21761|
    |          163|   20662|
    |          164|   19648|
    |          263|   18480|
    |          249|   18398|
    +-------------+--------+
    only showing top 20 rows
    
    The execution time is: 5.593951463699341
    Number of pickup hours: 260
    The speed per hour group is 0.021515197937305157
    

**Analyzing some financial records**


## **Ratio of the driver's compulsory road tolls to the money earned by the total customer**

Here the idea is to calculate the ratio of the driver's compulsory road tolls to the money earned by the total customer. 'select' function is executed to query the column 'tolls_amount' and the column 'total_amount' with nested function 'Round(NVL())' to create a new column based on the previous two column that are mentioned. The NVL function is used to replace NULL value with another value.



```python
 df.createOrReplaceTempView("taxi_trips") #creating a table.
sqlDF = spark.sql("SELECT * FROM taxi_trips")
start_time=time.time() 
sqlDF.show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
```

    +--------+--------------------+---------------------+---------------+-------------+----------+------------------+------------+------------+------------+-----------+-----+-------+----------+------------+---------------------+------------+
    |VendorID|tpep_pickup_datetime|tpep_dropoff_datetime|passenger_count|trip_distance|RatecodeID|store_and_fwd_flag|PULocationID|DOLocationID|payment_type|fare_amount|extra|mta_tax|tip_amount|tolls_amount|improvement_surcharge|total_amount|
    +--------+--------------------+---------------------+---------------+-------------+----------+------------------+------------+------------+------------+-----------+-----+-------+----------+------------+---------------------+------------+
    |       1| 2017-01-09 11:13:28|  2017-01-09 11:25:45|              1|          3.3|         1|                 N|         263|         161|           1|       12.5|  0.0|    0.5|       2.0|         0.0|                  0.3|        15.3|
    |       1| 2017-01-09 11:32:27|  2017-01-09 11:36:01|              1|          0.9|         1|                 N|         186|         234|           1|        5.0|  0.0|    0.5|      1.45|         0.0|                  0.3|        7.25|
    |       1| 2017-01-09 11:38:20|  2017-01-09 11:42:05|              1|          1.1|         1|                 N|         164|         161|           1|        5.5|  0.0|    0.5|       1.0|         0.0|                  0.3|         7.3|
    |       1| 2017-01-09 11:52:13|  2017-01-09 11:57:36|              1|          1.1|         1|                 N|         236|          75|           1|        6.0|  0.0|    0.5|       1.7|         0.0|                  0.3|         8.5|
    |       2| 2017-01-01 00:00:00|  2017-01-01 00:00:00|              1|         0.02|         2|                 N|         249|         234|           2|       52.0|  0.0|    0.5|       0.0|         0.0|                  0.3|        52.8|
    |       1| 2017-01-01 00:00:02|  2017-01-01 00:03:50|              1|          0.5|         1|                 N|          48|          48|           2|        4.0|  0.5|    0.5|       0.0|         0.0|                  0.3|         5.3|
    |       2| 2017-01-01 00:00:02|  2017-01-01 00:39:22|              4|         7.75|         1|                 N|         186|          36|           1|       22.0|  0.5|    0.5|      4.66|         0.0|                  0.3|       27.96|
    |       1| 2017-01-01 00:00:03|  2017-01-01 00:06:58|              1|          0.8|         1|                 N|         162|         161|           1|        6.0|  0.5|    0.5|      1.45|         0.0|                  0.3|        8.75|
    |       1| 2017-01-01 00:00:05|  2017-01-01 00:08:33|              2|          0.9|         1|                 N|          48|          50|           1|        7.0|  0.5|    0.5|       0.0|         0.0|                  0.3|         8.3|
    |       2| 2017-01-01 00:00:05|  2017-01-01 00:05:04|              5|         1.76|         1|                 N|         140|          74|           2|        7.0|  0.5|    0.5|       0.0|         0.0|                  0.3|         8.3|
    |       2| 2017-01-01 00:00:05|  2017-01-01 00:15:36|              1|         8.47|         1|                 N|         138|         262|           1|       24.0|  0.5|    0.5|      7.71|        5.54|                  0.3|       38.55|
    |       1| 2017-01-01 00:00:06|  2017-01-01 00:11:56|              2|          2.4|         1|                 N|         142|         236|           2|       10.5|  0.5|    0.5|       0.0|         0.0|                  0.3|        11.8|
    |       1| 2017-01-01 00:00:06|  2017-01-01 00:23:37|              2|         12.6|         5|                 N|         161|         265|           1|       60.0|  0.0|    0.0|      10.0|         0.0|                  0.3|        70.3|
    |       1| 2017-01-01 00:00:06|  2017-01-01 00:08:53|              1|          0.9|         1|                 N|         234|         186|           1|        7.0|  0.5|    0.5|      2.05|         0.0|                  0.3|       10.35|
    |       2| 2017-01-01 00:00:06|  2017-01-01 00:09:30|              4|         2.43|         1|                 N|         141|         107|           1|        9.5|  0.5|    0.5|       2.7|         0.0|                  0.3|        13.5|
    |       2| 2017-01-01 00:00:06|  2017-01-01 00:16:05|              2|          2.6|         1|                 N|          79|         163|           1|       12.5|  0.5|    0.5|      2.76|         0.0|                  0.3|       16.56|
    |       2| 2017-01-01 00:00:06|  2017-01-01 00:18:12|              5|         4.25|         1|                 N|         148|          36|           2|       16.5|  0.5|    0.5|       0.0|         0.0|                  0.3|        17.8|
    |       2| 2017-01-01 00:00:07|  2017-01-01 00:07:42|              1|         0.65|         1|                 N|          48|          68|           1|        6.5|  0.5|    0.5|       1.7|         0.0|                  0.3|         9.5|
    |       2| 2017-01-01 00:00:09|  2017-01-01 00:34:21|              1|         3.42|         1|                 N|         230|         148|           1|       22.5|  0.5|    0.5|       0.0|         0.0|                  0.3|        23.8|
    |       1| 2017-01-01 00:00:10|  2017-01-01 00:24:52|              1|          6.6|         1|                 N|         186|         232|           2|       23.0|  0.5|    0.5|       0.0|         0.0|                  0.3|        24.3|
    +--------+--------------------+---------------------+---------------+-------------+----------+------------------+------------+------------+------------+-----------+-----+-------+----------+------------+---------------------+------------+
    only showing top 20 rows
    
    The execution time is: 0.10923504829406738
    


```python
start_time=time.time()
spark.sql( "SELECT tolls_amount, total_amount, ROUND(NVL((tolls_amount / total_amount)*100,0),2) AS EXP_RATE from taxi_trips WHERE NVL((tolls_amount / total_amount)*100,0) <> 0 ORDER BY EXP_RATE DESC").show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
```

    +------------+------------+--------+
    |tolls_amount|total_amount|EXP_RATE|
    +------------+------------+--------+
    |        12.5|        12.8|   97.66|
    |        12.5|        20.8|    60.1|
    |        5.54|       11.34|   48.85|
    |        12.5|       26.41|   47.33|
    |        10.5|        22.3|   47.09|
    |        25.0|        55.8|    44.8|
    |        20.0|        47.3|   42.28|
    |        25.0|        60.8|   41.12|
    |        10.5|        25.8|    40.7|
    |        10.5|        28.3|    37.1|
    |        25.0|        67.8|   36.87|
    |        12.5|        34.3|   36.44|
    |        10.5|        32.3|   32.51|
    |        5.54|       17.21|   32.19|
    |        5.54|       17.34|   31.95|
    |        12.5|        39.8|   31.41|
    |       16.62|       54.42|   30.54|
    |        5.54|       18.39|   30.13|
    |        5.54|       18.84|   29.41|
    |        12.5|        43.3|   28.87|
    +------------+------------+--------+
    only showing top 20 rows
    
    The execution time is: 0.34723591804504395
    

## **Trips more than 15 miles long and trip's financial data**


```python
var4 = df.select(col("trip_distance"), col("tip_amount"), col("tolls_amount"), col("total_amount"),col("payment_type"),col("fare_amount"), col("extra"), col("mta_tax"), col("improvement_surcharge")).\
where("trip_distance > 15")
var4.orderBy("trip_distance","total_amount","tolls_amount", ascending=False).\
show(10, truncate = False)

```

    +-------------+----------+------------+------------+------------+-----------+-----+-------+---------------------+
    |trip_distance|tip_amount|tolls_amount|total_amount|payment_type|fare_amount|extra|mta_tax|improvement_surcharge|
    +-------------+----------+------------+------------+------------+-----------+-----+-------+---------------------+
    |151.7        |0.0       |0.0         |550.3       |2           |550.0      |0.0  |0.0    |0.3                  |
    |139.17       |0.0       |0.0         |350.8       |2           |350.0      |0.0  |0.5    |0.3                  |
    |120.6        |15.0      |5.54        |240.84      |1           |220.0      |0.0  |0.0    |0.3                  |
    |88.05        |0.0       |5.54        |306.34      |2           |300.0      |0.0  |0.5    |0.3                  |
    |86.3         |10.0      |29.04       |279.34      |1           |240.0      |0.0  |0.0    |0.3                  |
    |80.9         |0.0       |0.0         |52.8        |3           |52.0       |0.0  |0.5    |0.3                  |
    |79.0         |0.0       |5.54        |5.85        |2           |0.01       |0.0  |0.0    |0.3                  |
    |76.94        |0.0       |0.0         |125.3       |2           |125.0      |0.0  |0.0    |0.3                  |
    |76.12        |0.0       |5.54        |205.84      |2           |200.0      |0.0  |0.0    |0.3                  |
    |73.8         |0.0       |5.54        |156.34      |2           |150.0      |0.0  |0.5    |0.3                  |
    +-------------+----------+------------+------------+------------+-----------+-----+-------+---------------------+
    only showing top 10 rows
    
    

## **Trips with trip distance greater than the average trip distance**


```python
var5 = spark.sql("select PULocationID ||'-'|| DOLocationID AS ROAD,trip_distance, Fare_amount, Extra, Total_amount from taxi_trips group by ROAD,trip_distance, Fare_amount, Extra, Total_amount having trip_distance > avg(Trip_distance) order by trip_distance desc")
start = time.time()
var5.show()
end = time.time()
print(f"Execution time: {end - start}")
```

    +-------+-------------+-----------+-----+------------+
    |   ROAD|trip_distance|Fare_amount|Extra|Total_amount|
    +-------+-------------+-----------+-----+------------+
    |132-243|         21.4|       52.0|  0.0|       69.99|
    |132-166|         18.6|       52.0|  0.0|       69.99|
    |132-230|         18.1|       52.0|  0.0|       58.34|
    |132-230|         17.8|       52.0|  0.0|       58.34|
    |132-107|         17.6|       52.0|  0.0|       69.99|
    |230-132|         17.3|       52.0|  0.0|       58.34|
    |230-132|         17.1|       52.0|  0.0|       58.34|
    |138-230|         11.7|       36.0|  0.0|       50.79|
    |138-162|         10.7|       30.5|  0.5|       44.79|
    | 161-66|          7.1|       24.0|  0.0|        24.8|
    | 161-13|          6.6|       21.0|  0.0|        21.8|
    |264-264|          3.8|       14.5|  0.5|        15.8|
    |264-264|          3.8|       16.5|  0.0|        17.3|
    |231-100|          3.8|       16.0|  0.0|        16.8|
    | 79-263|          3.8|       13.5|  0.0|        14.3|
    |264-264|          3.8|       15.5|  0.0|        16.3|
    |264-264|          3.8|       15.5|  0.5|        16.8|
    |164-263|          3.3|       12.0|  0.5|       15.95|
    | 79-140|          3.3|       12.0|  0.5|       15.95|
    |186-236|          3.3|       14.5|  0.0|       18.35|
    +-------+-------------+-----------+-----+------------+
    only showing top 20 rows
    
    Execution time: 8.424556255340576
    

## **Which payment method is the most preferred? What is the max amount paid by customer?**


In this question, the exact defifition of which payment method is the most preferred, that is, which payment method appears most frequently in the dataset. 'case... when ...' statement is used to answer the question. The CASE statement goes through conditions and returns a value when the first condition is met.So once a condition is true, it will stop reading and return the result. If no conditions are true, it returns the value in the ELSE clause. Since in the dataset, all the values of the column 'payment_type' are numbers,thus all payment types are renamed in case when statement. Meanwhile, a new column with the name "payment_type_usage" is added to present the count of each payment type and grouped by this new column. To find the max amount paid by customer of each payment type, MAX() function is used.









```python
start_time=time.time()
spark.sql("SELECT case when payment_type= 1 then 'Credit card' when payment_type=2 then 'Cash' when payment_type= 3 then 'No charge'  when payment_type= 4 then 'Dispute' when payment_type= 5 then 'Unknown' when payment_type= 6 then 'Voided Trip' else payment_type end as Desc_Payment_Types, COUNT(*) as payment_type_usage, MAX(total_amount) FROM taxi_trips group by payment_type order by payment_type_usage desc").show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
```

    +------------------+------------------+-----------------+
    |Desc_Payment_Types|payment_type_usage|max(total_amount)|
    +------------------+------------------+-----------------+
    |       Credit card|             18532|            300.3|
    |              Cash|             17392|           166.34|
    |         No charge|               204|            150.0|
    |           Dispute|                57|            83.84|
    |              null|                 1|             null|
    +------------------+------------------+-----------------+
    
    The execution time is: 1.7614789009094238
    

## **What is the most 5 common travels?**

Here, the values of column 'PULocationID' and the column 'DOLocationID' concatenated with '-' and created a new column named 'ROAD' which contains the concatenated values and used 'count' function to count the number of the absolute frequency of each value of the new column 'ROAD'. As a last step, the RDD sorted by number of trips in descending order.



```python
#The most 5 common travels.
df_com_1 = df.select(
              col("PULocationID"),
              col("DOLocationID"),
              concat( col("PULocationID"),lit("-"),col("DOLocationID")).alias("ROAD")
)
df_common = df_com_1.filter("PULocationID <> DOLocationID").groupBy("ROAD").agg(count('ROAD').alias("Num of Trips")).sort(desc('Num of Trips'))
start_time=time.time()
data_amount = df_common.count()
df_common.show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
print('Number of pickup hours: {}'.format(data_amount))
print('The speed per hour group is {}'.format((end_time-start_time)/data_amount))
```

    +-------+------------+
    |   ROAD|Num of Trips|
    +-------+------------+
    |237-236|        3433|
    |236-237|        3011|
    |230-246|        2944|
    |239-142|        2595|
    |239-238|        2539|
    |142-239|        2471|
    |230-186|        2206|
    | 79-107|        2104|
    |238-239|        2085|
    |186-230|        2084|
    |  48-68|        1994|
    |263-141|        1887|
    |142-238|        1868|
    | 79-170|        1852|
    | 148-79|        1819|
    | 249-79|        1811|
    |107-170|        1807|
    |141-236|        1777|
    | 48-246|        1742|
    |186-170|        1700|
    +-------+------------+
    only showing top 20 rows
    
    The execution time is: 7.233892917633057
    Number of pickup hours: 16516
    The speed per hour group is 0.000437993032067877
    

## **What is the pickpoint zones and average total amount, tips and the maximum number of passangers?**

To answer this question, the new lookup table 'taxi+_zone_lookup.csv' was used.
"join" function was used to join the two datasets 'tripdata_2017-01.csv' and 'taxi+_zone_lookup.csv' together. 'left join' function was used which returned all records from the left table ('tripdata_2017-01.csv'), and the matching records from the right table ('taxi+_zone_lookup.csv').









```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").\
master("local[*]").getOrCreate()

df_dim = spark.read.format("csv").\
load("taxi+_zone_lookup.csv",inferSchema =True,sep=",",header="true") #read the csv file
df_dim.show()
```

    +----------+-------------+--------------------+------------+
    |LocationID|      Borough|                Zone|service_zone|
    +----------+-------------+--------------------+------------+
    |         1|          EWR|      Newark Airport|         EWR|
    |         2|       Queens|         Jamaica Bay|   Boro Zone|
    |         3|        Bronx|Allerton/Pelham G...|   Boro Zone|
    |         4|    Manhattan|       Alphabet City| Yellow Zone|
    |         5|Staten Island|       Arden Heights|   Boro Zone|
    |         6|Staten Island|Arrochar/Fort Wad...|   Boro Zone|
    |         7|       Queens|             Astoria|   Boro Zone|
    |         8|       Queens|        Astoria Park|   Boro Zone|
    |         9|       Queens|          Auburndale|   Boro Zone|
    |        10|       Queens|        Baisley Park|   Boro Zone|
    |        11|     Brooklyn|          Bath Beach|   Boro Zone|
    |        12|    Manhattan|        Battery Park| Yellow Zone|
    |        13|    Manhattan|   Battery Park City| Yellow Zone|
    |        14|     Brooklyn|           Bay Ridge|   Boro Zone|
    |        15|       Queens|Bay Terrace/Fort ...|   Boro Zone|
    |        16|       Queens|             Bayside|   Boro Zone|
    |        17|     Brooklyn|             Bedford|   Boro Zone|
    |        18|        Bronx|        Bedford Park|   Boro Zone|
    |        19|       Queens|           Bellerose|   Boro Zone|
    |        20|        Bronx|             Belmont|   Boro Zone|
    +----------+-------------+--------------------+------------+
    only showing top 20 rows
    
    

**joining tables**


```python
df_join = df.join(df_dim,df.PULocationID ==  df_dim.LocationID,"left")
df_join.show()

```

    +--------+--------------------+---------------------+---------------+-------------+----------+------------------+------------+------------+------------+-----------+-----+-------+----------+------------+---------------------+------------+----------+---------+--------------------+------------+
    |VendorID|tpep_pickup_datetime|tpep_dropoff_datetime|passenger_count|trip_distance|RatecodeID|store_and_fwd_flag|PULocationID|DOLocationID|payment_type|fare_amount|extra|mta_tax|tip_amount|tolls_amount|improvement_surcharge|total_amount|LocationID|  Borough|                Zone|service_zone|
    +--------+--------------------+---------------------+---------------+-------------+----------+------------------+------------+------------+------------+-----------+-----+-------+----------+------------+---------------------+------------+----------+---------+--------------------+------------+
    |       1| 2017-01-09 11:13:28|  2017-01-09 11:25:45|              1|          3.3|         1|                 N|         263|         161|           1|       12.5|  0.0|    0.5|       2.0|         0.0|                  0.3|        15.3|       263|Manhattan|      Yorkville West| Yellow Zone|
    |       1| 2017-01-09 11:32:27|  2017-01-09 11:36:01|              1|          0.9|         1|                 N|         186|         234|           1|        5.0|  0.0|    0.5|      1.45|         0.0|                  0.3|        7.25|       186|Manhattan|Penn Station/Madi...| Yellow Zone|
    |       1| 2017-01-09 11:38:20|  2017-01-09 11:42:05|              1|          1.1|         1|                 N|         164|         161|           1|        5.5|  0.0|    0.5|       1.0|         0.0|                  0.3|         7.3|       164|Manhattan|       Midtown South| Yellow Zone|
    |       1| 2017-01-09 11:52:13|  2017-01-09 11:57:36|              1|          1.1|         1|                 N|         236|          75|           1|        6.0|  0.0|    0.5|       1.7|         0.0|                  0.3|         8.5|       236|Manhattan|Upper East Side N...| Yellow Zone|
    |       2| 2017-01-01 00:00:00|  2017-01-01 00:00:00|              1|         0.02|         2|                 N|         249|         234|           2|       52.0|  0.0|    0.5|       0.0|         0.0|                  0.3|        52.8|       249|Manhattan|        West Village| Yellow Zone|
    |       1| 2017-01-01 00:00:02|  2017-01-01 00:03:50|              1|          0.5|         1|                 N|          48|          48|           2|        4.0|  0.5|    0.5|       0.0|         0.0|                  0.3|         5.3|        48|Manhattan|        Clinton East| Yellow Zone|
    |       2| 2017-01-01 00:00:02|  2017-01-01 00:39:22|              4|         7.75|         1|                 N|         186|          36|           1|       22.0|  0.5|    0.5|      4.66|         0.0|                  0.3|       27.96|       186|Manhattan|Penn Station/Madi...| Yellow Zone|
    |       1| 2017-01-01 00:00:03|  2017-01-01 00:06:58|              1|          0.8|         1|                 N|         162|         161|           1|        6.0|  0.5|    0.5|      1.45|         0.0|                  0.3|        8.75|       162|Manhattan|        Midtown East| Yellow Zone|
    |       1| 2017-01-01 00:00:05|  2017-01-01 00:08:33|              2|          0.9|         1|                 N|          48|          50|           1|        7.0|  0.5|    0.5|       0.0|         0.0|                  0.3|         8.3|        48|Manhattan|        Clinton East| Yellow Zone|
    |       2| 2017-01-01 00:00:05|  2017-01-01 00:05:04|              5|         1.76|         1|                 N|         140|          74|           2|        7.0|  0.5|    0.5|       0.0|         0.0|                  0.3|         8.3|       140|Manhattan|     Lenox Hill East| Yellow Zone|
    |       2| 2017-01-01 00:00:05|  2017-01-01 00:15:36|              1|         8.47|         1|                 N|         138|         262|           1|       24.0|  0.5|    0.5|      7.71|        5.54|                  0.3|       38.55|       138|   Queens|   LaGuardia Airport|    Airports|
    |       1| 2017-01-01 00:00:06|  2017-01-01 00:11:56|              2|          2.4|         1|                 N|         142|         236|           2|       10.5|  0.5|    0.5|       0.0|         0.0|                  0.3|        11.8|       142|Manhattan| Lincoln Square East| Yellow Zone|
    |       1| 2017-01-01 00:00:06|  2017-01-01 00:23:37|              2|         12.6|         5|                 N|         161|         265|           1|       60.0|  0.0|    0.0|      10.0|         0.0|                  0.3|        70.3|       161|Manhattan|      Midtown Center| Yellow Zone|
    |       1| 2017-01-01 00:00:06|  2017-01-01 00:08:53|              1|          0.9|         1|                 N|         234|         186|           1|        7.0|  0.5|    0.5|      2.05|         0.0|                  0.3|       10.35|       234|Manhattan|            Union Sq| Yellow Zone|
    |       2| 2017-01-01 00:00:06|  2017-01-01 00:09:30|              4|         2.43|         1|                 N|         141|         107|           1|        9.5|  0.5|    0.5|       2.7|         0.0|                  0.3|        13.5|       141|Manhattan|     Lenox Hill West| Yellow Zone|
    |       2| 2017-01-01 00:00:06|  2017-01-01 00:16:05|              2|          2.6|         1|                 N|          79|         163|           1|       12.5|  0.5|    0.5|      2.76|         0.0|                  0.3|       16.56|        79|Manhattan|        East Village| Yellow Zone|
    |       2| 2017-01-01 00:00:06|  2017-01-01 00:18:12|              5|         4.25|         1|                 N|         148|          36|           2|       16.5|  0.5|    0.5|       0.0|         0.0|                  0.3|        17.8|       148|Manhattan|     Lower East Side| Yellow Zone|
    |       2| 2017-01-01 00:00:07|  2017-01-01 00:07:42|              1|         0.65|         1|                 N|          48|          68|           1|        6.5|  0.5|    0.5|       1.7|         0.0|                  0.3|         9.5|        48|Manhattan|        Clinton East| Yellow Zone|
    |       2| 2017-01-01 00:00:09|  2017-01-01 00:34:21|              1|         3.42|         1|                 N|         230|         148|           1|       22.5|  0.5|    0.5|       0.0|         0.0|                  0.3|        23.8|       230|Manhattan|Times Sq/Theatre ...| Yellow Zone|
    |       1| 2017-01-01 00:00:10|  2017-01-01 00:24:52|              1|          6.6|         1|                 N|         186|         232|           2|       23.0|  0.5|    0.5|       0.0|         0.0|                  0.3|        24.3|       186|Manhattan|Penn Station/Madi...| Yellow Zone|
    +--------+--------------------+---------------------+---------------+-------------+----------+------------------+------------+------------+------------+-----------+-----+-------+----------+------------+---------------------+------------+----------+---------+--------------------+------------+
    only showing top 20 rows
    
    

Here, from the new table which was joined by the previous two tables, several columns are queried and gaved them alias. Especially, the hour() function is used to extract the hour part from timestamp data of the column 'tpep_pickup_datetime'.


```python
df_1 = df_join.select(     
                           col("PULocationID").alias("Pickup_ID"),
                           col("Borough"), 
                           col("Zone").alias("Pickup_Zone"),
                           col("passenger_count").alias("num_of_passangers"),
                           col("tip_amount"),
                           col("total_amount"),
                           hour(col("tpep_pickup_datetime")).alias("pickup_hour"),
                           col("extra") )
start_time = time.time()
data_amount = df_1.count()
df_1.printSchema()
df_1.show()
end_time = time.time()
print('The execution time is: {}'.format(end_time-start_time))
print('Number of pickup hours: {}'.format(data_amount))
print('The speed per hour group is {}'.format((end_time-start_time)/data_amount))

```

    root
     |-- Pickup_ID: integer (nullable = true)
     |-- Borough: string (nullable = true)
     |-- Pickup_Zone: string (nullable = true)
     |-- num_of_passangers: integer (nullable = true)
     |-- tip_amount: double (nullable = true)
     |-- total_amount: double (nullable = true)
     |-- pickup_hour: integer (nullable = true)
     |-- extra: double (nullable = true)
    
    +---------+---------+--------------------+-----------------+----------+------------+-----------+-----+
    |Pickup_ID|  Borough|         Pickup_Zone|num_of_passangers|tip_amount|total_amount|pickup_hour|extra|
    +---------+---------+--------------------+-----------------+----------+------------+-----------+-----+
    |      263|Manhattan|      Yorkville West|                1|       2.0|        15.3|         11|  0.0|
    |      186|Manhattan|Penn Station/Madi...|                1|      1.45|        7.25|         11|  0.0|
    |      164|Manhattan|       Midtown South|                1|       1.0|         7.3|         11|  0.0|
    |      236|Manhattan|Upper East Side N...|                1|       1.7|         8.5|         11|  0.0|
    |      249|Manhattan|        West Village|                1|       0.0|        52.8|          0|  0.0|
    |       48|Manhattan|        Clinton East|                1|       0.0|         5.3|          0|  0.5|
    |      186|Manhattan|Penn Station/Madi...|                4|      4.66|       27.96|          0|  0.5|
    |      162|Manhattan|        Midtown East|                1|      1.45|        8.75|          0|  0.5|
    |       48|Manhattan|        Clinton East|                2|       0.0|         8.3|          0|  0.5|
    |      140|Manhattan|     Lenox Hill East|                5|       0.0|         8.3|          0|  0.5|
    |      138|   Queens|   LaGuardia Airport|                1|      7.71|       38.55|          0|  0.5|
    |      142|Manhattan| Lincoln Square East|                2|       0.0|        11.8|          0|  0.5|
    |      161|Manhattan|      Midtown Center|                2|      10.0|        70.3|          0|  0.0|
    |      234|Manhattan|            Union Sq|                1|      2.05|       10.35|          0|  0.5|
    |      141|Manhattan|     Lenox Hill West|                4|       2.7|        13.5|          0|  0.5|
    |       79|Manhattan|        East Village|                2|      2.76|       16.56|          0|  0.5|
    |      148|Manhattan|     Lower East Side|                5|       0.0|        17.8|          0|  0.5|
    |       48|Manhattan|        Clinton East|                1|       1.7|         9.5|          0|  0.5|
    |      230|Manhattan|Times Sq/Theatre ...|                1|       0.0|        23.8|          0|  0.5|
    |      186|Manhattan|Penn Station/Madi...|                1|       0.0|        24.3|          0|  0.5|
    +---------+---------+--------------------+-----------------+----------+------------+-----------+-----+
    only showing top 20 rows
    
    The execution time is: 2.1757853031158447
    Number of pickup hours: 971010
    The speed per hour group is 2.240744485757968e-06
    


```python
df_1.createOrReplaceTempView("taxi_trips_zone")
df_2=spark.sql("SELECT Borough, round(AVG(total_amount),2) as avg_total_amount, max(num_of_passangers), round(avg(tip_amount),2) as avg_tip_amount from taxi_trips_zone group by Borough order by avg_total_amount desc")
start_time=time.time()
data_amount = df_2.count()
df_2.show()
end_time=time.time()
print('The execution time is: {}'.format(end_time-start_time))
print('Number of pickup hours: {}'.format(data_amount))
print('The speed per hour group is {}'.format((end_time-start_time)/data_amount))

```

    +-------------+----------------+----------------------+--------------+
    |      Borough|avg_total_amount|max(num_of_passangers)|avg_tip_amount|
    +-------------+----------------+----------------------+--------------+
    |          EWR|           82.71|                     4|         12.51|
    |       Queens|           42.18|                     9|          4.91|
    |Staten Island|           40.72|                     5|           2.8|
    |      Unknown|           18.17|                     9|          1.94|
    |     Brooklyn|           16.91|                     6|          1.76|
    |        Bronx|           15.95|                     6|          0.78|
    |    Manhattan|           14.15|                     9|          1.44|
    +-------------+----------------+----------------------+--------------+
    
    The execution time is: 6.505506753921509
    Number of pickup hours: 7
    The speed per hour group is 0.9293581077030727
    
