# Datasets

Flight Delay and Cancellation. 

## Description

Airline Flight Delay and Cancellation Data, August 2019 - August 2023. 
US Department of Transportation, Bureau of Transportation Statistics
https://www.transtats.bts.gov

## Purpose

Analyzing flight delays, cancellations, and their causes. 
Studying airline and airport performance. 
Predictive modeling for delays. 
How air lines group support can work to improve these delay factors to shorter the delay time and reduce flight cancellations. 

## Dataset location
[Git URL](https://github.com/gaurav1der/ml-ai-examples/submission/1.4/datasets)

## Summary of the characteristics of the dataset

### General Overview:

Type: Flight records (U.S. domestic flights). 
Rows: Several hundred (sample shown, actual may be larger). 
Columns: 31. 

### Key Columns:

Flight Info:  
    FL_DATE (flight date), AIRLINE, AIRLINE_CODE, FL_NUMBER. 
Airports:  
    ORIGIN, ORIGIN_CITY, DEST, DEST_CITY. 
Times:  
    Scheduled and actual departure/arrival times (CRS_DEP_TIME, DEP_TIME, CRS_ARR_TIME, ARR_TIME). 
    Delays (DEP_DELAY, ARR_DELAY). 
    Taxi times (TAXI_OUT, TAXI_IN). 
    Wheels off/on times. 
Flight Status:  
    CANCELLED, CANCELLATION_CODE, DIVERTED. 
Elapsed Times:  
    Scheduled and actual elapsed time, air time, distance. 
Delay Causes:  
    Delays due to carrier, weather, NAS, security, late aircraft. 

# Datasets

Traffic congestion. 

## Description

Traffic congestion and related problems are a common concern in urban areas. Understanding traffic patterns and analyzing data can provide valuable insights for transportation planning, infrastructure development, and congestion management.  

## Purpose

Analyzing traffic patterns by time of day and day of week. 
Predicting traffic situations based on vehicle counts. 
Studying the distribution of different vehicle types.  

## Dataset location
[Git URL](https://github.com/gaurav1der/ml-ai-examples/submission/1.4/datasets)

## Summary of the characteristics of the dataset

### General Overview:

Type: Traffic count and situation data. 
Rows: Several hundred (covers multiple days and time intervals). 
Columns: 9. 

### Key Columns:

Time: Time of observation (e.g., 12:00:00 AM). 
Date: Day of the month (numeric). 
Day of the week: Day name (e.g., Tuesday). 
CarCount: Number of cars observed. 
BikeCount: Number of bikes observed. 
BusCount: Number of buses observed. 
TruckCount: Number of trucks observed. 
Total: Total vehicle count (sum of above). 
Traffic Situation: Categorical label (e.g., low, normal, high, heavy) indicating traffic condition. 
