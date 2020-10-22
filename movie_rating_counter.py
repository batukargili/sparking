from pyspark import SparkConf, SparkContext
import collections


def ratings_counter():
    """
    Data ml-100k/u.data downloaded from https://grouplens.org/datasets/movielens/.
    Download the data and specify the path.

    :return: Rating Counts as OrderedDict
    """
    conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")  # SparkConf
    sc = SparkContext(conf=conf)  # SparkContext
    lines = sc.textFile("/home/ozer/SparkCourse/ml-100k/u.data")  # specify your own path
    ratings = lines.map(lambda x: x.split()[2])  # PipelinedRDD
    result = ratings.countByValue()  # collections.defaultdict
    sorted_results = collections.OrderedDict(sorted(result.items()))  # collection.OrderedDict

    for key, value in sorted_results.items():
        print("%s %i" % (key, value))
    return sorted_results


ratings_counter()
