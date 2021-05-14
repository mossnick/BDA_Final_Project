from pyspark import SparkContext
import datetime
from datetime import timedelta
import csv
import functools
import json
import numpy as np
import sys
import math

 
def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]
    group_dict = {'big_box_grocers':0,'convenience_stores':1,'drinking_places':2,'full_service_restaurants':3,'limited_service_restaurants':4,'pharmacies_and_drug_stores':5,
              'snack_and_bakeries':6,'specialty_food_stores':7,'supermarkets_except_convenience_stores':8}
    # Step C
    CAT_CODES = set(['445210', '445110', '722410', '452311', '722513', '445120', '446110', '445299', '722515', '311811', '722511', '445230', '446191', '445291', '445220', '452210', '445292'])
    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446110': 5, '446191': 5, '722515': 6, '311811': 6, '445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7, '445110': 8}

    
    #Step D
    def filterPOIs(_, lines):    
        reader = csv.reader(lines)
        for row in reader:
            if row[9] in CAT_CODES:
                yield (row[0],CAT_GROUP[row[9]])
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs).cache()
    
    # Step E
    storeGroup = dict(rddD.collect())
    print(storeGroup['23g-222@627-wc8-7h5']) # for sanity check, should be 6
    groupCount = rddD \
        .map(lambda x: (x[1],1)) \
        .reduceByKey(lambda x,y: x+y) \
        .sortByKey() \
        .map(lambda x: x[1]) \
        .collect()
        
    # Step G
    def extractVisits(storeGroup, partId, lines):
        if partId == 0:
            next(lines)
        import csv
        rows = csv.reader(lines)
        for row in rows:
            year = row[12][:4]
            placekey = row[0]
            start_day = row[12][:10]
            visits = json.loads(row[16])
            if placekey in storeGroup:
                for i in range(0,7):
                    date_obj = datetime.strptime(start_day,"%Y-%m-%d") + timedelta(days=i)
                    if date_obj.year in [2019,2020]:
                        date_diff = date_obj - datetime.strptime("2019-01-01","%Y-%m-%d")
                        yield ( (storeGroup[placekey],date_diff.days),visits[i])
    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))
    
    # Step H
    def custom_std(data, ddof=0):
        n = len(data)
        mean = sum(data) / n
        v = sum((x - mean) ** 2 for x in data) / (n - ddof)
        return math.sqrt(v) 
    def computeStats(groupCount, kv):  
        values = list(kv[1])
        group = kv[0][0]
        total_visits = groupCount[group]
        expand_values_by = total_visits - len(values)
        values = [0]*expand_values_by + values
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        median = (sorted_values[mid] + sorted_values[~mid]) / 2
        std = custom_std(sorted_values)
        low = max(0,median-std)
        hi = median + std
        return (kv[0],(low,median,hi))    

    rddH = rddG.groupByKey() \
            .map(functools.partial(computeStats, groupCount))

    
    # Step I
    def pretty_output(rec):
        current_date = datetime.strptime("2019-01-01","%Y-%m-%d") + timedelta(days=rec[0][1])
        year = current_date.year
        current_date_str = "2020" + current_date.strftime("%Y-%m-%d")[4:]
        return (rec[0][0],"{4},{0},{1},{2},{3}".format(current_date_str,round(rec[1][0]),round(rec[1][1]),round(rec[1][2]),year))
    rddI = rddH \
            .map(pretty_output)
    

    # Step J
    rddJ = rddI.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()
    # rddJ.take(5)


    # Step K
    for group_name,group_num in group_dict.items():
        filename = group_name
        rddJ.filter(lambda x: x[0]==group_num or x[0]==-1).values() \
            .saveAsTextFile(f'{OUTPUT_PREFIX}/{filename}')
 
if __name__=='__main__':
    sc = SparkContext()
    main(sc)