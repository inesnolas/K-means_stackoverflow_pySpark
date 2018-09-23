# _______________________________________________________________________________________________
# to run in windows use
# rmdir c:\tmp\ /S /Q
# cd c:\ProgramData\hadoop-2.8.1\etc\hadoop
# hadoop-env.cmd
# %HADOOP_PREFIX%\bin\hdfs namenode -format
# %HADOOP_PREFIX%\sbin\start-dfs.cmd

# cd c:\ProgramData\hadoop-2.8.1
# %HADOOP_PREFIX%\bin\hdfs dfs -put myfile.txt /
# cd C:\Users\eurico\WordDocuments\QMUL_MSc\BigData\coursework2\pyspark
# REM python get_all_user_variables.py
# %HADOOP_PREFIX%\bin\hdfs dfs -put userDirectFeatures.csv /
# %HADOOP_PREFIX%\bin\hdfs dfs -ls /

# %HADOOP_PREFIX%\sbin\start-yarn.cmd
# cd c:\ProgramData\hadoop-2.8.1\etc\hadoop
# hadoop-env.cmd
# cd c:\ProgramData\hadoop-2.8.1\
# hdfs dfs -ls /
# mkdir C:\tmp\hive
# %HADOOP_HOME%\bin\winutils.exe chmod 777 /tmp/hive
# cd c:\PROGRA~1\Anaconda3\Scripts
# %SPARK_HOME%\bin\load-spark-env.cmd
# cd C:\Users\eurico\WordDocuments\QMUL_MSc\BigData\coursework2\pyspark
# hdfs dfs -ls /
# hdfs dfs -mkdir /user
# hdfs dfs -mkdir /user/group-AI
# hdfs dfs -put output_user_clean.csv /user/group-AI/
# hdfs dfs -ls /user/group-AI/
# C:\ProgramData\spark-2.1.2-bin-hadoop2.7\bin\spark-submit --master local[*] Kmeans_ForStackFile.py --k 2 --epsilon 0.001 --maxiter 25 --fileportion 0.2
# _______________________________________________________________________________________________
# to run on linux cluster use
# spark-submit --master yarn --packages com.databricks:spark-csv_2.10:1.4.0 Kmeans_ForStackFile.py --k 2 --epsilon 0.001 --maxiter 25 --fileportion 0.2
# _______________________________________________________________________________________________
from numpy import *
import csv
import datetime
from pyspark import SparkContext

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--k", help="use k clusters.")
parser.add_argument("--epsilon", help="use epsilon of cost delta to exit.")
parser.add_argument("--maxiter", help="use max iterations.")
parser.add_argument("--fileportion", help="use percentage or file portion of file.")
args = parser.parse_args()
if args.k:
    k = int(args.k)
else:
    k = 2

if args.epsilon:
    epsilon = float(args.epsilon)
else:
    epsilon = 0.001

if args.maxiter:
    maxiter = int(args.maxiter)
else:
    maxiter = 25

if args.fileportion:
    fileportion = float(args.fileportion)
else:
    fileportion = 0.2

arguments_in="k="+str(k)+"_epsilon="+str(epsilon)+"_maxiter="+str(maxiter)+"_fileportion="+str(fileportion)
arguments_in.replace(" ", "").replace("\t", "")

#print("k=",k," epsilon=",epsilon," maxiter=",maxiter," fileportion=",fileportion)
print("k=",k," epsilon=",epsilon," fileportion=",fileportion)

def compute_newCentroids(sum_coord, count):
    c=[]
    for x in sum_coord:
        t=x/count
        c.append(t)
    return c

def create_dict(table):
    dict_table={}
    for t in table:
        dict_table[t[0]]=t[-1]
    return dict_table


def calculate_closest_centroid(row, dict_centroids, k):
    dist = 100000000000    #if this i not big enough k+1 clusters will be created!! verify this somehow!!
    index = k+1
    for i in range(k):
        aa=array(row)
        bb = array(dict_centroids[i])
        new_distance=linalg.norm(aa-bb)**2
        if new_distance < dist:
            dist = new_distance
            index = i
    return (index, dist)

def parsePoint_fromString(st):
    point=[float(j) for j in st.split(",")]
    return point

def parsePoint_fromStringArray(st):
    point=[float(j) for j in st]
    return point

print("settings up clustering sc variable")
sc = SparkContext(appName = "Clustering")  # initialize the sc 'spark context'for  spark.
# for some reasdon this works in pyspark but not in spark-submit, so we need to initialize explicitly spark sql
print("settings up SQLContext spark variable")
from pyspark.sql import SQLContext
spark = SQLContext(sc)
#Creating the RDD File
print("creating rdd to file")
# file = sc.textFile("/user/group-AI/output_user_clean.csv")
# myfile = spark.read.option("header","true").csv("/user/group-AI/output_user_clean.csv").rdd.map(tuple)
myfile = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/user/group-AI/output_user_clean.csv").rdd.map(tuple)
#print("removing header")
#file_withoutHeader=file.filter(lambda row: row.find('reputation')!=0)
# CHANGE COMMENTed lines LINE TO TAKE ONLY A SUPSET OF POINTS:+++++++++++++++++++++++++++++++++++++++++++
print("taking small sample")
rdd_part=myfile.sample(False, fileportion, 0) # change this number to % you want, 1 means full file! 7,5 million rows, 2 GB!!!
#rdd_file=file.rdd.map(lambda row:  parsePoint_fromString(row) )
rdd_part_parsed=rdd_part.map(lambda row:  parsePoint_fromStringArray(row) )
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("going to cache")
rdd_part_parsed.cache()
count_points=rdd_part_parsed.count()
print("Number of Points being used=", count_points)




delta_cost=10000.0
list_cost=[]
cost=100000
rseed=0
print("epsilon of delta_cost = ", epsilon)

print("CENTROIDs INITIALIZATION")
#CENTROIDs INITIALIZATION:
iniCentroids = rdd_part_parsed.takeSample(False, k, rseed)
#verification to make sure we dont have repeated centroids!!!
mset = [list(x) for x in set(tuple(x) for x in iniCentroids)]
while (len(iniCentroids) is not len(mset)):
    print("will take another sample1")
    rseed+=1
#           #take new samples until length are the same!
    iniCentroids = rdd_part_parsed.takeSample(False,k, rseed)
    mset = [list(x) for x in set(tuple(x) for x in iniCentroids)]

print("Print CENTROIDs table and dictionary")
centroids_table=[]  #Create dictionary of centroids from inicentroids
for c in range(k):
    centroids_table.append((c, iniCentroids[c]))
    print(centroids_table[c])


dict_centroids=create_dict(centroids_table)   # hashtable of centroids {0: [x, y, z], 1:[, , , ]...}

outputFile=open("inputkmeans_"+arguments_in+".csv", "w")
writer = csv.writer(outputFile)
for key, value in dict_centroids.items():
    writer.writerow([key, value])

itr=0

print("Initial cost is _______-> ", cost)
print("Initial Delta is ______ -> ", delta_cost)

# KMEANS CYCLE:
#while ((delta_cost>epsilon) and (itr < maxiter) ):

while (delta_cost>epsilon):
    itr=itr+1
#DISTANCE COMPUTATION---Assignment RDD
    #Calculate closest centroid and add it to each point
#   rdd_centroids = rdd_file.map(lambda row : (row,calculate_closest_centroid(row,centroids, k)))
    rdd_centroids = rdd_part_parsed.map(lambda row : (row,calculate_closest_centroid(row,dict_centroids, k))).cache()


#COST COMPUTATION:
    cost_prev=cost

    rdd_dist= rdd_centroids.map(lambda row : (int(row[-1][0]),float(row[-1][1]))).reduceByKey(lambda a,b: (a+b))
    dist = rdd_dist.collect(); # dist= rdd_dist.reduce(lambda a ,b: a+b)
    cost=sum(x[1] for x in dist)/count_points  #cost=dist/count_points

#   print("COST IS ", cost)
    list_cost.append(cost)
    delta_cost=abs(cost_prev-cost)
#   print(" new DELTA IS ", delta_cost)

#UPDATE CENTROIDS:
    rdd_temp = rdd_centroids.map(lambda row : (int(row[-1][0]), row[0])).cache()

#   rdd_count = rdd_temp.map(lambda row : (row[0],1)).reduceByKey(lambda a,b : (a+b)).sortByKey()
    rdd_count = rdd_temp.map(lambda row : (row[0],1)).reduceByKey(lambda a,b : (a+b))
#   count=rdd_count.collect()
#   dict_count=create_dict(count)
    rdd_sum_computation = rdd_temp.reduceByKey(lambda a,b :[x + y for x, y in zip(a, b)])
    rdd_index_sum_counts = rdd_sum_computation.join(rdd_count) #[(0, [sum centroid0], count)..]
#   rdd_mean_computation = rdd_sum_computation.map(lambda row : (row[0],compute_newCentroids(row, count))).sortByKey()
#   print("rdd_index_sum_counts:  ", rdd_index_sum_counts.collect())
#   sys.exit("centroid update FINISHED!!!!!!!!!!!!!!1")
    rdd_centroids_update = rdd_index_sum_counts.map(lambda row : (row[0],compute_newCentroids(row[-1][0], row[-1][-1])))

    centroids=rdd_centroids_update.collect()
    dict_centroids=create_dict(centroids)
#   centroids=rdd_mean_computation.map(lambda row: (row[1])).collect()



#   if itr in arange(0, 10000, 10):
    print("K=",k,"itr=",itr,"cost=", cost, "delta_cost=", delta_cost)

print("OUTPUT CREATION")
# OUTPUT CREATION:

# only USE FOR SMALL SUBSET OF DATA!!!!!!!!!!! writes assignment table to file .> makes COLLECT()+++++++++++++++++++++++++++++++++++++++++++++++
# #
#rdd_assign=rdd_temp.sortByKey()
#assign=rdd_assign.collect()
#output_pontos=open("pontosTeste_" +str(nowsrt)+ ".csv", "w")
#writer = csv.writer(output_pontos)
#writer.writerows(assign)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# USE FOR WHOLE DATASET: writes assignment table to HDFS:++++++++++++++++++++++++++++++++++++++++
#
rdd_assign=rdd_temp.sortByKey()
st_path="/user/group-AI/output_points_K3"
rdd_assign.saveAsTextFile(st_path)
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

outputFile=open("outputkmeans_"+arguments_in+".csv", "w")
writer = csv.writer(outputFile)
for key, value in dict_centroids.items():
    writer.writerow([key, value])

print("list_cost=",list_cost)
outputCost=open("costoverIterations_"+arguments_in+".csv", "w")
outputCost.writelines(["%s\n" %item for item in list_cost])


