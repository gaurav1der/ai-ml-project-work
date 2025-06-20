# Datasets

Flight Delay and Cancellation 

## Description

Airline Flight Delay and Cancellation Data, August 2019 - August 2023

US Department of Transportation, Bureau of Transportation Statistics
https://www.transtats.bts.gov

## Purpose

✔ Analyzing flight delays, cancellations, and their causes

✔ Studying airline and airport performance

✔ Predictive modeling for delays

✓ How air lines group support can work to improve these delay factors to shorter the delay time and reduce flight cancellations

## Dataset location
[Git URL](https://github.com/gaurav1der/ml-ai-examples/submission/1.4/datasets)

## Summary of the characteristics of the dataset

### General Overview:

Type: Flight records (U.S. domestic flights)

Rows: Several hundred (sample shown, actual may be larger)

Columns: 31

### Key Columns:

Flight Info:

    FL_DATE (flight date), AIRLINE, AIRLINE_CODE, FL_NUMBER

Airports:

    ORIGIN, ORIGIN_CITY, DEST, DEST_CITY

Times:

    Scheduled and actual departure/arrival times (CRS_DEP_TIME, DEP_TIME, CRS_ARR_TIME, ARR_TIME)

    Delays (DEP_DELAY, ARR_DELAY)

    Taxi times (TAXI_OUT, TAXI_IN)

    Wheels off/on times

Flight Status:

    CANCELLED, CANCELLATION_CODE, DIVERTED

Elapsed Times:

    Scheduled and actual elapsed time, air time, distance

Delay Causes:

    Delays due to carrier, weather, NAS, security, late aircraft