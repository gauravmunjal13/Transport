
# coding: utf-8

# ### Get the Spark Context

# In[42]:

sc


# In[43]:

sc.applicationId


# ### Load the required packages

# In[3]:

import sys
sys.path.append("/usr/lib/python2.7/site-packages")

get_ipython().magic(u'matplotlib notebook')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.functions import col, asc, desc,log
from graphframes import *
import folium
import colour


# ### Load the VTS data

# In[4]:

sqlContext.sql("use bmtcvts")
## vts_df = sqlContext.sql("select * from vts_parse_data_parquet")
vts_df = sqlContext.sql("select * from vts_sept16_parquet")


# ### Load the static data

# In[5]:

sqlContext.sql("use bmtc")

# Get the route map
route_map_df = sqlContext.sql("select route_id,start_bus_stop_id,end_bus_stop_id,                                      distance,time_to_travel,bus_stop_order,status                                from route_map")

# Get the route_point
route_point_df = sqlContext.sql("select route_id, route_order, bus_stop_id from route_point")
bus_stop_df = sqlContext.sql("select bus_stop_id,bus_stop_name,latitude_current,longitude_current from bus_stop")
# Drop corrupted locations
bus_stop_df = bus_stop_df.na.drop(subset=["latitude_current"])
bus_stop_df = bus_stop_df.na.drop(subset=["longitude_current"])

# Join the bus stop ID with lat,long
route_point_joined_df = route_point_df.join(bus_stop_df,                                            ["bus_stop_id"],                                            "left_outer")

form_four_df = sqlContext.sql("select form_four_id,form_four_name,schedule_number_id,                                      schedule_number_name,no_of_trips,start_time,                                      route_id,route_number,toll_zone,                                      area_limit,total_km,total_dead_km,                                      actual_km,total_running_time,total_break_time,                                      total_steering_time,spread_over_hours,ot_hours                                from form_four")

schedule_df = sqlContext.sql('select * from schedule')
schedule_df = schedule_df.select("schedule_id","schedule_number","depot_id","route_id","schedule_type")
schedule_details_df = sqlContext.sql('select * from schedule_details')
schedule_details_df = schedule_details_df.select("schedule_details_id","form_four_id","schedule_number","number_of_trips",                           "trip_number","trip_type","start_point","end_point","route_number_id",                           "route_number","route_direction","distance","start_time","end_time",                           "running_time","break_type_id","shift_type_id","is_dread_trip")


# In[152]:

# Get the waybill details, and clean it
waybill_trip_details_df = sqlContext.sql("select id,waybill_id,duty_dt,device_id,                                          status,schedule_type_id,schedule_no,schedule_name,                                          service_type,service_name,trip_number,                                          start_point,start_bus_stop_name,end_point,end_bus_stop_name,                                          route_id,route_no,distance,start_time,                                          act_start_time,etm_start_time,end_time,act_end_time,                                          etm_end_time,running_time,is_dread_trip                                           from waybill_trip_details")

waybill_trip_details_filtered_df = waybill_trip_details_df.filter((((year(waybill_trip_details_df.duty_dt) == 2016) &                                                                     (month(waybill_trip_details_df.duty_dt) == 9)) &                                                                    (dayofmonth(waybill_trip_details_df.duty_dt) == 2))
                                                                  
                                                                  )

waybill_trip_details_valid_df = waybill_trip_details_filtered_df.where((col("status") != "NEW") &                                                                          (col("status") != "325-703-380") &                                                                          (col("status") != "325-703-3144") &                                                                          (col("status") != "325-703-560") &                                                                          (col("status") != "325-702-741") &                                                                          (col("status") != "325-702-835"))


# ### Clean the data

# In[71]:

# Get the route_ids from the waybill trip data
waybill_route_ids_list = waybill_trip_details_valid_df.select("route_id").distinct().rdd.map(lambda x:x[0]).collect()
print("Length of route_ids from waybill",len(waybill_route_ids_list))

# Out of the list of waybill route_ids, how many of them have route_maps
waybill_route_ids_with_route_map_list = route_map_df[route_map_df.route_id.isin(waybill_route_ids_list)]                                                    .select("route_id").distinct().rdd.map(lambda x:x[0]).collect()
print("Length of waybill route_ids having maps",len(waybill_route_ids_with_route_map_list))

# Get the list of route_ids from waybill, having no route map
waybill_route_ids_with_no_map_list = list(set(waybill_route_ids_list) - set(waybill_route_ids_with_route_map_list))
print("Length of waybill route_ids having no route map",len(waybill_route_ids_with_no_map_list))

# Find the entries in waybill table corresponding to waybill_route_ids_with_no_map
waybill_with_no_map_df = waybill_trip_details_valid_df[waybill_trip_details_valid_df.route_id.isin(waybill_route_ids_with_no_map_list)]
print("Length of entries in waybill with no route map",waybill_with_no_map_df.count())
print("Waybill route_ids with no route map",waybill_route_ids_with_no_map_list)

# Valid waybill table
waybill_with_map_df = waybill_trip_details_valid_df.where(~col("route_id").isin(waybill_route_ids_with_no_map_list))

# Depot bus stops 
depot_bus_stops_id_list = waybill_with_map_df.filter(col("is_dread_trip")==1).select("start_point").distinct().rdd.map(lambda x:x[0]).collect()


# In[72]:

# Get the route maps of those routes which are part of the waybill
route_point_joined_valid_df = route_point_joined_df.where(col("route_id").isin(waybill_route_ids_with_route_map_list))
route_map_valid_df = route_map_df.where(col("route_id").isin(waybill_route_ids_with_route_map_list))

# Get the valid form_four_ids
valid_schedules_list = waybill_with_map_df.select("schedule_no").distinct().rdd.map(lambda x:x[0]).collect()
form_four_valid_df = form_four_df.filter(col("form_four_id").isin(valid_schedules_list))


# ### Feature Engineering on VTS data

# In[73]:

# Remove the corrupted Lat and Long values
filtered_vts_df = vts_df.where((vts_df.lat != 0) & (vts_df.longitude != 0) & (vts_df.ign_status == 1))


# ### Explor schedule 335E

# In[74]:

# Get the list of all schedule names starting with 335
#schedule_335E = waybill_with_map_df.filter(col('schedule_name').like("%335E%")).select("schedule_name")\
#                   .distinct().rdd.map(lambda x:x[0]).collect()


# In[75]:

#schedule_335E


# #### V-335E/44-All Days

# In[ ]:

# Take schedule V-335E/44-All Days, and we see that it takes two schedule no, which corresponds to two form four IDs
#waybill_with_map_df.filter(col('schedule_name')=='V-335E/44-All Days').select("schedule_no").distinct().show()


# In[ ]:

#form_four_valid_df.filter(col('form_four_id').isin([12292])).show()


# In[ ]:

# Check the waybill data having schedule no listed above
#waybill_with_map_df.filter(col('schedule_no')==12292).select('schedule_no','trip_number',\
#                                                             'start_point','start_bus_stop_name',\
#                                                             'end_point','end_bus_stop_name',\
#                                                             'route_id','is_dread_trip').show()


# In[ ]:

# Get the schedule details
#schedule_details_df.filter((col("schedule_number")==5006) & (col('form_four_id')==12292) ).show()


# ## Trip time analysis for all schedules

# In[76]:

import datetime
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.functions import col,unix_timestamp,abs, from_unixtime, avg, count, sum, desc, date_format, lit, concat


# In[77]:

#scheduleNo = 12332
strYear="2016"
strMonth="09"
strDay="01"
dateLit=strYear+strMonth+strDay
dateHyphen=strYear+"-"+strMonth+"-"+strDay
monthLit= strMonth+strYear
queryYear = int(strYear)
queryMonth = int(strMonth)
queryDay = int(strDay)


# ## Filter waybill data by month:

# In[78]:

# Filter waybill trip details by month
# TODO: Dayofmonth removed
waybill_filtered_df = waybill_with_map_df.filter((year(waybill_with_map_df.etm_start_time) == queryYear) &                                                               (month(waybill_with_map_df.etm_start_time) == queryMonth))
# Filter further on schedule no
#waybill_filtered_df = waybill_filtered_df.filter(col("schedule_no") == scheduleNo)


# ### Time window for the trips

# In[79]:

#waybill_filtered_df.printSchema()


# In[80]:

#waybill_filtered_df.show(5)


# In[81]:

dateTimeFmt = "yyyy-MM-dd HH:mm:ss.S"
dateFmt = "yyyy-MM-dd"
timeFmt = "HH:mm:ss"
timeDiff = abs(unix_timestamp('end_time', format=timeFmt)            - unix_timestamp('start_time', format=timeFmt))
endTimeLimit = from_unixtime(unix_timestamp('end_time', format=timeFmt)                           + timeDiff, format=timeFmt)
etm_start_time_date_part = from_unixtime(unix_timestamp('etm_start_time', format=dateTimeFmt), format=dateFmt)
etm_start_time_timestamp_part = from_unixtime(unix_timestamp('etm_start_time', format=dateTimeFmt), format=timeFmt)
etm_end_time_timestamp_part = from_unixtime(unix_timestamp('etm_end_time', format=dateTimeFmt), format=timeFmt)
#dateTimeFmt1 = "yyyy-MM-dd"
#timestampFormat = "HH:mm:ss"


# Define the window limit as (startTime, (endTime + (endTime-startTime))).
# This is done to factor in any delays in trip completion. 
# Assumption is that a trip will be completed atleast in twice the time that it is supposed to take. 

# In[82]:

waybill_filtered_df = waybill_filtered_df.withColumn("end_time_limit", endTimeLimit)
waybill_filtered_df = waybill_filtered_df.withColumn("business_date", etm_start_time_date_part)
waybill_filtered_df = waybill_filtered_df.withColumn("etm_start_timestamp", etm_start_time_timestamp_part)
waybill_filtered_df = waybill_filtered_df.withColumn("etm_end_timestamp", etm_end_time_timestamp_part)


# In[83]:

# TODO: may need to select more columns: add start and end time since trip number is not correct
sqlContext.registerDataFrameAsTable(waybill_filtered_df, "way_bill_data_table")
windowQuery = """SELECT id, waybill_id, duty_dt, device_id, schedule_no, route_id, trip_number,
is_dread_trip, start_point, end_point, start_time, end_time,act_start_time, act_end_time, 
etm_start_timestamp, etm_end_timestamp,
concat(business_date," ",(CASE 
    WHEN (act_start_time == 'null' or act_start_time == '00:00:00'
        or act_end_time == 'null' or act_end_time == '00:00:00' or act_start_time > act_end_time) 
    THEN etm_start_timestamp 
    ELSE act_start_time 
END)) as window_start_time,
concat(business_date," ",(CASE 
    WHEN (act_start_time == 'null' or act_start_time == '00:00:00'
        or act_end_time == 'null' or act_end_time == '00:00:00' or act_start_time > act_end_time) 
    THEN etm_end_timestamp 
    ELSE act_end_time
END)) as window_end_time
FROM way_bill_data_table"""
waybill_filtered_sample_df = sqlContext.sql(windowQuery)


# In[84]:

#waybill_filtered_sample_df.printSchema()


# In[85]:

# TODO: Extension format and hardcore value for the extension
windowStartTimeFmt = "yyyy-MM-dd HH:mm:ss"
windowExtension = abs(unix_timestamp('window_end_time', format=windowStartTimeFmt)            - unix_timestamp('window_start_time', format=windowStartTimeFmt))
windowExtensionFunction = from_unixtime(unix_timestamp("window_end_time", format=windowStartTimeFmt)                           + windowExtension, format=windowStartTimeFmt)


# In[86]:

#waybill_filtered_sample_df.select("waybill_id","start_time","end_time","act_start_time","act_end_time",
#                                  "etm_start_timestamp","etm_end_timestamp","window_start_time","window_end_time").show(2)


# In[87]:

waybill_filtered_sample_df = waybill_filtered_sample_df.withColumn("window_end_time",windowExtensionFunction)


# In[88]:

#waybill_filtered_sample_df.select("waybill_id","start_time","end_time","act_start_time","act_end_time",
#                                  "etm_start_timestamp","etm_end_timestamp","window_start_time","window_end_time").show(2)


# Filter VTS data by month:

# In[153]:

vts_sample_df = filtered_vts_df.filter((year(filtered_vts_df.ist_date) == queryYear) &                                      (month(filtered_vts_df.ist_date) == queryMonth) &                                      (dayofmonth(filtered_vts_df.ist_date) == 2))


# In[90]:

#waybill_filtered_sample_df.show(10)


# In[91]:

#route_point_joined_valid_df.printSchema()


# In[92]:

#waybill_filtered_sample_df.printSchema()


# In[93]:

waybill_route_point_joined_start_sample_df = waybill_filtered_sample_df.withColumnRenamed("route_id","route_id_waybill")
waybill_route_point_joined_start_sample_df = waybill_route_point_joined_start_sample_df.withColumnRenamed("device_id","device_id_waybill")
# Add the column to identify the start bus stop
waybill_route_point_joined_start_sample_df = waybill_route_point_joined_start_sample_df.join(route_point_joined_valid_df,
        ((waybill_route_point_joined_start_sample_df.route_id_waybill == route_point_joined_valid_df.route_id)
        & (waybill_route_point_joined_start_sample_df.start_point == route_point_joined_valid_df.bus_stop_id)),"left_outer")\
        .select("device_id_waybill", "schedule_no", "trip_number", "route_id_waybill",\
        "window_start_time", "window_end_time", "is_dread_trip", "start_point", "end_point", "route_order")\
        .withColumnRenamed("route_order","start_route_order") 
        


# In[94]:

# Get the column to provide the bus stop order corresponding to the end route point
waybill_route_point_joined_start_end_sample_df = waybill_route_point_joined_start_sample_df.join(route_point_joined_valid_df, 
                ((waybill_route_point_joined_start_sample_df.route_id_waybill == route_point_joined_valid_df.route_id)
                 & (waybill_route_point_joined_start_sample_df.end_point == route_point_joined_valid_df.bus_stop_id)),"left_outer")\
                .select("device_id_waybill", "schedule_no", "trip_number", "route_id_waybill",\
                "window_start_time", "window_end_time", "is_dread_trip", "start_point", "end_point",\
                "start_route_order", "route_order")\
                .withColumnRenamed("route_order","end_route_order") 


# In[95]:

waybill_device_id_route_id_bus_stop_id_df = waybill_route_point_joined_start_end_sample_df.join(route_point_joined_valid_df,               (waybill_route_point_joined_start_end_sample_df.route_id_waybill == route_point_joined_valid_df.route_id),               "left_outer")


# In[96]:

#waybill_device_id_route_id_bus_stop_id_df.show(10)


# In[97]:

waybill_device_id_route_id_bus_stop_id_df = waybill_device_id_route_id_bus_stop_id_df                                                .filter((col("route_order") >= col("start_route_order"))                                                 & (col("route_order") <= col("end_route_order")))


# In[98]:

waybill_device_id_bus_stop_id_df = waybill_device_id_route_id_bus_stop_id_df.select(    "device_id_waybill", "schedule_no","trip_number", "route_id_waybill", "bus_stop_id", "route_order",             "window_start_time", "window_end_time", "is_dread_trip", "start_route_order", "end_route_order")


# In[99]:

waybill_device_id_bus_stop_id_df = waybill_device_id_bus_stop_id_df.withColumnRenamed("bus_stop_id", "route_bus_stop_id")


# ### Populate distance and time_to_travel from route_map table

# In[100]:

# Populate distance and time_to_travel as 0 for start_bus_stop in a route
route_start_bus_stop_df = waybill_device_id_bus_stop_id_df.where(col("route_order") == col("start_route_order"))
route_start_bus_stop_df = route_start_bus_stop_df.withColumn("distance", lit(0))
route_start_bus_stop_df = route_start_bus_stop_df.withColumn("time_to_travel", lit(0))


# In[101]:

# Joining with Route_Map for distance and time_to_travel
route_rest_bus_stop_df = waybill_device_id_bus_stop_id_df.where(col("route_order") != col("start_route_order"))


# In[102]:

mapped_route_rest_bus_stop_df = route_rest_bus_stop_df.join(route_map_df, 
                                          ((route_rest_bus_stop_df.route_id_waybill == route_map_df.route_id)\
                                           & (route_rest_bus_stop_df.route_bus_stop_id == route_map_df.end_bus_stop_id)),\
                                           "left_outer") \
                                .drop("route_id").drop("start_bus_stop_id").drop("end_bus_stop_id").drop("bus_stop_order").drop("status")


# In[103]:

#route_map_df.printSchema()


# In[104]:

#route_start_bus_stop_df.printSchema()


# In[105]:

#route_rest_bus_stop_df.printSchema()


# In[106]:

resultant_waybill_device_id_bus_stop_id_df = route_start_bus_stop_df.unionAll(mapped_route_rest_bus_stop_df)


# In[107]:

#resultant_waybill_device_id_bus_stop_id_df.show(5)


# ###  Attach the bus stop lat long values with device id-bus stop by joining with bus stop table

# In[108]:

#bus_stop_df.printSchema()


# In[109]:

resultant_waybill_device_id_bus_stop_id_mapped_df =  resultant_waybill_device_id_bus_stop_id_df.join(bus_stop_df,                             resultant_waybill_device_id_bus_stop_id_df.route_bus_stop_id == bus_stop_df.bus_stop_id,                             "left_outer").drop("bus_stop_id").drop("start_route_order").drop("end_route_order")


# In[110]:

#resultant_waybill_device_id_bus_stop_id_mapped_df.printSchema()


# In[111]:

# Remove the dead trips for the analysis
resultant_waybill_mapped_no_dead_trip_df = resultant_waybill_device_id_bus_stop_id_mapped_df.where(col("is_dread_trip") ==0)


# In[112]:

timeFmt_Ist = "yyyy-MM-dd HH:mm:ss.S"
timeFmt1 = "yyyy-MM-dd HH:mm:ss"
istDateTrunc=from_unixtime(unix_timestamp('ist_date', format=timeFmt_Ist), format=timeFmt1)


# In[113]:

vts_sample_df = vts_sample_df.withColumn("ist_timestamp", istDateTrunc)


# ### Join the resultant waybill table with the vts_parse table

# In[114]:

# TODO: for distinct
resultant_waybill_mapped_no_dead_trip_df = resultant_waybill_mapped_no_dead_trip_df.distinct()
vts_waybill_df = resultant_waybill_mapped_no_dead_trip_df.join(vts_sample_df,
                        resultant_waybill_mapped_no_dead_trip_df.device_id_waybill == vts_sample_df.device_id,
                        "left_outer")


# ### Geo-Fencing

# In[115]:

sqlContext.registerDataFrameAsTable(vts_waybill_df, "vts_Table")
query = """SELECT device_id_waybill,schedule_no, trip_number, route_id_waybill, route_bus_stop_id, route_order, 
window_start_time, window_end_time, ist_date, ist_timestamp, acc_distance, 
(((acos(sin((latitude_current*pi()/180)) * 
sin((lat*pi()/180))+cos((latitude_current*pi()/180)) * 
cos((lat*pi()/180)) * cos(((longitude_current- longitude)* 
pi()/180))))*180/pi())*40*1.1515)*1000 as dist 
FROM vts_Table where ist_timestamp >= window_start_time and ist_timestamp <= window_end_time
HAVING (dist > 0 AND dist <= 40) order by IST_DATE,route_bus_stop_id"""


# In[154]:

sqlContext.sql(query).coalesce(1).write.option("header", "true")    .csv("mka/bmtc/Day2/All_Schedule/windowed_times/")


# In[117]:

###sqlContext.sql(query).coalesce(1).write.option("header", "true")\
###    .csv("mka/bmtc/Day1/"+monthLit+"/All_Schedule/windowed_bus_stop_geo_fence_distances/1-sep")


# ### Post Processing after geo-fencing

# In[119]:

# Read the saved geo-fence distance data from HDFS
windowGeoFenceDf = sqlContext.read.option("header", "true")                .option("inferSchema", "true")                .csv("mka/bmtc/Day2/All_Schedule/windowed_times")


# In[120]:

#windowGeoFenceDf.count()


# In[121]:

#windowGeoFenceDf.printSchema()


# In[122]:

windowGeoFenceDf.select("trip_number","route_bus_stop_id","route_order","ist_timestamp","dist").show(10, truncate = False)


# In[123]:

# Finding the earliest entry into a bus stop geo-fence by a device

windowGeoFenceDf = windowGeoFenceDf.withColumn("ist_timestamp_seconds", unix_timestamp("ist_timestamp", format=timeFmt1))

windowGeoFenceDf = windowGeoFenceDf.withColumnRenamed("device_id_waybill", "device_id")
windowGeoFenceDf = windowGeoFenceDf.withColumnRenamed("route_id_waybill", "route_id")

windowEarliestGeoFenceEntryDf = windowGeoFenceDf.groupby("device_id", "schedule_no", "trip_number", "route_id", "route_bus_stop_id","route_order", "window_start_time", "window_end_time").min("ist_timestamp_seconds")                              .withColumnRenamed("min(ist_timestamp_seconds)", "ist_timestamp_seconds")


# In[124]:

windowEarliestGeoFenceEntryDf.printSchema()


# In[125]:

windowEarliestGeoFenceEntryDf.select("device_id","trip_number","route_bus_stop_id","route_order","ist_timestamp_seconds").show(10, truncate = False)


# ### Populating the distance travelled by device when it enters the geo-fence of a bus stop

# In[126]:

windowGeoFenceDf= windowGeoFenceDf.withColumnRenamed("device_id", "device_id_d").withColumnRenamed("route_id", "route_id_d").withColumnRenamed("schedule_no", "schedule_no_d").withColumnRenamed("trip_number", "trip_number_d").withColumnRenamed("route_bus_stop_id", "route_bus_stop_id_d").withColumnRenamed("route_order", "route_order_d").withColumnRenamed("window_start_time", "window_start_time_d").withColumnRenamed("window_end_time", "window_end_time_d").withColumnRenamed("ist_timestamp_seconds", "ist_timestamp_seconds_d")


# In[127]:

earliestGeoFenceEntryWithDistanceDf= windowEarliestGeoFenceEntryDf.join(windowGeoFenceDf, ((windowEarliestGeoFenceEntryDf.device_id == windowGeoFenceDf.device_id_d)                           & (windowEarliestGeoFenceEntryDf.route_id == windowGeoFenceDf.route_id_d)                           & (windowEarliestGeoFenceEntryDf.schedule_no == windowGeoFenceDf.schedule_no_d)                           & (windowEarliestGeoFenceEntryDf.trip_number == windowGeoFenceDf.trip_number_d)                           & (windowEarliestGeoFenceEntryDf.route_bus_stop_id == windowGeoFenceDf.route_bus_stop_id_d)                           & (windowEarliestGeoFenceEntryDf.route_order == windowGeoFenceDf.route_order_d)                           & (windowEarliestGeoFenceEntryDf.window_start_time == windowGeoFenceDf.window_start_time_d)                           & (windowEarliestGeoFenceEntryDf.window_end_time == windowGeoFenceDf.window_end_time_d)                           & (windowEarliestGeoFenceEntryDf.ist_timestamp_seconds == windowGeoFenceDf.ist_timestamp_seconds_d)))


# In[128]:

earliestGeoFenceEntryWithDistanceDf = earliestGeoFenceEntryWithDistanceDf.groupBy("device_id", "schedule_no", "trip_number", "route_id", "route_bus_stop_id", "route_order", "window_start_time","window_end_time", "ist_timestamp_seconds").min("acc_distance")


# In[129]:

earliestGeoFenceEntryWithDistanceDf.printSchema()


# In[130]:

earliestGeoFenceEntryWithDistanceDf = earliestGeoFenceEntryWithDistanceDf.withColumnRenamed("min(acc_distance)", "acc_distance")


# ### Enriching with useful columns

# In[131]:

earliestGeoFenceEntryWithDistanceDf = earliestGeoFenceEntryWithDistanceDf.withColumn("arrival_time", from_unixtime("ist_timestamp_seconds", format=timeFmt1))


# ### Joining with all trips data to find missing values

# In[132]:

earliestGeoFenceEntryWithDistanceDf.printSchema()


# In[133]:

earliestGeoFenceEntryWithDistanceDf=earliestGeoFenceEntryWithDistanceDf.withColumn("window_start_timestamp",             from_unixtime(unix_timestamp('window_start_time', format=dateTimeFmt),format="yyyy-MM-dd HH:mm:ss"))

earliestGeoFenceEntryWithDistanceDf=earliestGeoFenceEntryWithDistanceDf.withColumn("window_end_timestamp",             from_unixtime(unix_timestamp('window_end_time', format=dateTimeFmt),format="yyyy-MM-dd HH:mm:ss"))

earliestGeoFenceEntryWithDistanceDf=earliestGeoFenceEntryWithDistanceDf.drop("window_start_time").drop("window_end_time")

earliestGeoFenceEntryWithDistanceDf=earliestGeoFenceEntryWithDistanceDf.withColumnRenamed("window_start_timestamp","window_start_time").withColumnRenamed("window_end_timestamp","window_end_time")


# In[134]:

earliestGeoFenceEntryWithDistanceDf.count()


# In[135]:

#earliestGeoFenceEntryWithDistanceDf.show(5)


# ### Left outer join with waybill data (bus_stop_with_dread_trip_df) so as to ensure we identify missing values

# In[136]:

#resultant_waybill_device_id_bus_stop_id_mapped_df.show(5)
#resultant_waybill_mapped_no_dead_trip_df.show(5)
bus_stop_with_dead_trip_df = resultant_waybill_device_id_bus_stop_id_mapped_df.withColumnRenamed("schedule_no", "schedule_no_waybill").withColumnRenamed("trip_number", "trip_number_waybill").withColumnRenamed("route_bus_stop_id", "route_bus_stop_id_waybill").withColumnRenamed("route_order", "route_order_waybill").withColumnRenamed("window_start_time", "window_start_time_waybill").withColumnRenamed("window_end_time", "window_end_time_waybill").withColumnRenamed("latitude_current", "bus_stop_latitude").withColumnRenamed("longitude_current", "bus_stop_longitude")

bus_stop_with_dead_trip_df = bus_stop_with_dead_trip_df.withColumn("dayofweek", date_format("window_start_time_waybill", "EEEE"))
bus_stop_with_dead_trip_df = bus_stop_with_dead_trip_df.withColumn("month", lit(queryMonth))
bus_stop_with_dead_trip_df = bus_stop_with_dead_trip_df.withColumn("year", lit(queryYear))


# In[137]:

#bus_stop_with_dead_trip_df.printSchema()


# In[138]:

earliestGeoFenceEntryWithDistanceDf.printSchema()


# In[139]:

resultantEarliestGeoFenceEntryDf = bus_stop_with_dead_trip_df.join(earliestGeoFenceEntryWithDistanceDf,        ((bus_stop_with_dead_trip_df.device_id_waybill == earliestGeoFenceEntryWithDistanceDf.device_id)       & (bus_stop_with_dead_trip_df.route_id_waybill == earliestGeoFenceEntryWithDistanceDf.route_id)       & (bus_stop_with_dead_trip_df.schedule_no_waybill == earliestGeoFenceEntryWithDistanceDf.schedule_no)       & (bus_stop_with_dead_trip_df.trip_number_waybill == earliestGeoFenceEntryWithDistanceDf.trip_number)       & (bus_stop_with_dead_trip_df.route_bus_stop_id_waybill == earliestGeoFenceEntryWithDistanceDf.route_bus_stop_id)       & (bus_stop_with_dead_trip_df.route_order_waybill == earliestGeoFenceEntryWithDistanceDf.route_order)       & (bus_stop_with_dead_trip_df.window_start_time_waybill == earliestGeoFenceEntryWithDistanceDf.window_start_time)       & (bus_stop_with_dead_trip_df.window_end_time_waybill == earliestGeoFenceEntryWithDistanceDf.window_end_time)),        "left_outer")


# In[140]:

resultantEarliestGeoFenceEntryDf= resultantEarliestGeoFenceEntryDf.drop("device_id").drop("schedule_no").drop("trip_number").drop("route_id").drop("route_bus_stop_id").drop("route_order").drop("window_start_time").drop("window_end_time")


# In[141]:

resultantEarliestGeoFenceEntryDf= resultantEarliestGeoFenceEntryDf.withColumnRenamed("device_id_waybill", "device_id").withColumnRenamed("schedule_no_waybill", "schedule_no").withColumnRenamed("trip_number_waybill", "trip_number").withColumnRenamed("route_id_waybill", "route_id").withColumnRenamed("route_bus_stop_id_waybill", "route_bus_stop_id").withColumnRenamed("route_order_waybill", "route_order").withColumnRenamed("window_start_time_waybill", "window_start_time").withColumnRenamed("window_end_time_waybill", "window_end_time")


# In[142]:

resultantEarliestGeoFenceEntryDf.printSchema()


# In[143]:

resultantEarliestGeoFenceEntryDf= resultantEarliestGeoFenceEntryDf.withColumnRenamed("trip_number", "results_trip_number")


# ### Populating the results against schedule_details 

# In[144]:

schedule_details_filtered_df = schedule_details_df.select("form_four_id", "route_number_id", "trip_number", "start_time", "end_time")


# In[145]:

resultantEarliestGeoFenceEntryDf.printSchema()


# In[146]:

schedule_geo_entry_df = schedule_details_filtered_df.join(resultantEarliestGeoFenceEntryDf, 
        ((schedule_details_filtered_df.form_four_id == resultantEarliestGeoFenceEntryDf.schedule_no)\
        & (schedule_details_filtered_df.trip_number == resultantEarliestGeoFenceEntryDf.results_trip_number)
        & (schedule_details_filtered_df.route_number_id == resultantEarliestGeoFenceEntryDf.route_id)))

schedule_geo_entry_df = schedule_geo_entry_df.drop("schedule_no").drop("results_trip_number").drop("route_id")
schedule_geo_entry_df = schedule_geo_entry_df.withColumnRenamed("form_four_id", "schedule_no")
schedule_geo_entry_df = schedule_geo_entry_df.withColumnRenamed("route_number_id", "route_id")
schedule_geo_entry_df = schedule_geo_entry_df.withColumnRenamed("start_time", "scheduled_trip_start_time")
schedule_geo_entry_df = schedule_geo_entry_df.withColumnRenamed("end_time", "scheduled_trip_end_time")


# In[147]:

schedule_geo_entry_ordered_df = schedule_geo_entry_df.orderBy("schedule_no", "trip_number", "route_order")


# In[148]:

schedule_geo_entry_ordered_df.printSchema()


# In[150]:

schedule_geo_entry_ordered_df.coalesce(1).write.option("header","true")    .csv("mka/bmtc/Day2/All_Schedule/windowed_times/Featured")


# In[151]:

get_ipython().system(u'jupyter nbconvert --to script AllSchedule-InForms.ipynb')


# In[ ]:



