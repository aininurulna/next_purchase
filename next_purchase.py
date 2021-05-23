#Random Forest Regressor


#import the dataset
data = spark.read.csv("dbfs:/FileStore/tables/automotive_sales.csv", inferSchema = True, header = True, sep = ",").cache()

#check the data type of each feature
data.printSchema()

#change the new_purchase_date in case it's not date type yet
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_date
data = data.withColumn('new_purchase_date', to_date(unix_timestamp('purchase_date', 'yyyy/MM/dd').cast("timestamp"))).drop('purchase_date')

#sort by cust and purchase date
data = data.orderBy(["cust", "new_purchase_date"], ascending=True)


#calculating the days interval between purchases for each customer which will be our target variable
from pyspark.sql.functions import *
from pyspark.sql.window import Window
data = data.withColumn("timeInterval", datediff(data.new_purchase_date, lag(data.new_purchase_date, 1)
    .over(Window.partitionBy("cust")
    .orderBy("new_purchase_date"))))
data = data.na.fill({'timeInterval': 0})

#create flag of the i-th number of purchase (1st time, 2nd time, etc.) as a new feature
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, dense_rank
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_date

window = Window.partitionBy(data['cust']).orderBy(data['new_purchase_date'])
data = data.select('*', rank().over(window).alias('flag'))

#evaluate the correlation of numerical features with the target variable
import six

corr_cols = ["age"]
for i in corr_cols:
    if not( isinstance(data.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to timeInterval for ", i, data.stat.corr('timeInterval',i))
        
#build a pipeline to encode the categorical features

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
stages = []
cat_cols = ["gender", "product", "flag"]

for cat_col in cat_cols:
  indexer = StringIndexer(inputCol = cat_col, outputCol = cat_col + "_index", stringOrderType = "alphabetAsc")
  encoder = OneHotEncoder(inputCols = [indexer.getOutputCol()], outputCols = [cat_col + "_vec"], dropLast = True)
  stages += [indexer, encoder]
  
pipeline = Pipeline(stages=stages)
data2 = pipeline.fit(data).transform(data)

#split the date into training and testing with set up seed for consistency
(train, test) = data2.randomSplit([0.8, 0.2], seed=123)
train.cache()
test.cache()

# build a pipeline of VectorAssembler and rfr
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler

assemblerInputs = ["gender_vec", "product_vec", "flag_vec", "age"]

vectorAssembler = VectorAssembler(
  inputCols = assemblerInputs, 
  outputCol="features")

rfr = (RandomForestRegressor()
      .setLabelCol("timeInterval") # The column of our label
      .setSeed(123)        # Some seed value for consistency
      .setNumTrees(10)   # A guess at the number of trees
      .setMaxDepth(30)    # A guess at the depth of each tree
)

pipelinerfr = Pipeline().setStages([
  vectorAssembler,
  rfr
])

pipelineModelrfr = pipelinerfr.fit(train)

predictionsDFrfr = pipelineModelrfr.transform(test)

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator().setLabelCol("timeInterval")

rmserfr = evaluator.evaluate(predictionsDFrfr)

print("Test RMSE Random Forest Regressor = %f" % rmserfr)

#compare the RMSE with the average to get the sense of how well the model works as a simple analysis
from pyspark.sql.functions import col, avg
data.agg(avg(col("timeInterval"))).show()




