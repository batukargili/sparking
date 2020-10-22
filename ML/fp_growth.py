from datetime import datetime

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.fpm import FPGrowth, os
from pyspark.sql import functions as F
from pyspark.sql.functions import *
import pandas as pd
import numpy as np
import pymongo


def fp_growth_calc():
    df = data_enr()

    conf = SparkConf().setMaster("local").setAppName("FP_Growth")
    sc = SparkContext.getOrCreate(conf=conf)
    sqlContext = SQLContext(sc)

    sdf = sqlContext.createDataFrame(df.astype(str))
    new_sdf = sdf.withColumn("event_path", translate(F.col("event_path"), "[]' ", ""))
    splitted_sdf = new_sdf.withColumn("event_path", split(F.col("event_path"), ","))
    fpGrowth = FPGrowth(itemsCol="event_path", minSupport=0.5, minConfidence=0.4)
    model = fpGrowth.fit(splitted_sdf)
    result = model.freqItemsets.collect()
    result = result[0:10]
    test_json = {}
    count = 0
    local_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_json[str(local_dt)] = {"0": "", "1": "", "2": ""}
    for res in result:
        test_json[str(local_dt)][str(count)] = {"items": "", "freq": ""}
        test_json[str(local_dt)][str(count)]["items"] = res.asDict()['items']
        test_json[str(local_dt)][str(count)]["freq"] = res.asDict()['freq']
        count = count + 1
    return test_json


def data_enr():
    pd.options.display.max_columns = None

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['events']

    collection = db['transaction_events']

    data = pd.DataFrame(list(collection.find()))
    data_zeros = data.replace('--', '', regex=True)
    data_zeros['date'] = pd.to_datetime(data_zeros['date_client']).dt.date

    grouped_data = data_zeros.groupby(['django_cookie'], as_index=False)

    df = grouped_data.apply(lambda x: x)

    g_data = df.groupby(['django_cookie'])

    col_names = ['session_id', 'date', 'customer_id', 'customer_type', 'event_path', 'latitude', 'longitude',
                 'event_types',
                 'click_count', 'error_path', 'session_duration_sec']
    my_df2 = pd.DataFrame(columns=col_names)

    def remove_duplicates(l):
        return list(set(l))

    def indexCol(col):
        for i, li in enumerate(col):
            li = li.split(",")
            li = [x for x in li if x != '']

            for j, path in enumerate(li):
                len_path = len(path)
                count = 0
                for words in li[:j]:

                    if words[0:len(words) - 1] == path:
                        count = count + 1
                path = path + str(count)
                li[j] = path
            li = remove_duplicates(li)
            col[i] = li

        return col

    for name, group in g_data:
        cols = np.array(group.columns)
        errorListCase = (group.values == True)
        click_count = (group[group["eventType"] == "click"].count()["django_cookie"])
        session_id = name
        event_path = [','.join(group['eventPath'])]
        error_path = [','.join(cols[(row_index)]) for row_index in errorListCase]
        event_types = [','.join(group['eventType'])]
        longitude = group['longitude'].iloc[0]
        latitude = group['latitude'].iloc[0]
        date = pd.to_datetime(group['date_client'], unit='s').iloc[0]
        customer_id = group['customerId'].iloc[0]
        customer_type = group['customerType'].iloc[0]
        end = pd.to_datetime(group['django_cookie_end_time'], unit='s').max()
        start = pd.to_datetime(group['django_cookie_start_time'], unit='s').iloc[0]
        session_duration = pd.Timedelta(end - start).seconds

        event_path = indexCol(event_path)

        for i, li in enumerate(error_path):
            li = li.split(", ")
            li = [x for x in li if x != '']

        my_dic = {
            'session_id': session_id,
            'date': date,
            'customer_id': customer_id,
            'customer_type': customer_type,
            'event_path': event_path,
            'latitude': latitude,
            'longitude': longitude,
            'event_types': event_types,
            'click_count': click_count,
            'error_path': error_path,
            'session_duration_sec': session_duration,
        }

        my_df2.loc[len(my_df2)] = my_dic
    return my_df2
