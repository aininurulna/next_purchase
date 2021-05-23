# next_purchase
Predicting Next Purchase with PySpark

A Random Forest Classifier implemented in PySpark. The spyspark.ml.regression library was used to import the RandomForestRegressor. 
pyspark.ml.regression. The following arguments was passed to the object:

n_estimators = 10
max_depth = 30

Before the modelling, a pipeline was created to doone-hot encoding on the features. Since there are some string type features, we begin by using StringIndexer to first index the features and we then pass the results to OneHotEncoder which runs one-hot encoding on the features and stores the results as a sparse matrix, encoded using vectortype.

After that, another pipeline which contains VectorAssembler and RandomForestRegressor is created.
VectorAssembler: Assemble the feature columns into a feature vector.
