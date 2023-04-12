---
layout: post
title:  "Distributed training and test in Spark XGBoost"
published: true
mathjax: true
date:   2023-03-04 18:30:13 -0400
categories: default
tags: [Machine Learning, Distributed training, Spark, XGBoost]
---

Recently, we had done a project with xgboost model for classification. With the increasing of large amouts of data, we need to use XGBoost distributed training to replace the current pandas XGBoost training solution in Spark.

I explored the XGBoost training and test in Spark to record the framework here.

(1) Add the libraries.
from sparkxgb.xgboost import XGBoostClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

(2) Create spark conf environment for your app.
```
spark = SparkSession.builder..appname("test").getOrCreate()
```

(3) Read the data from gcs bucket or BQ
```
train_data_path = "train"
df_train = spark.read.parquet(data_path)

test_data_path = "test"
df_test = spark.read.parquet(test_data_path)
```

(4) Train on the model without fune tuning (assuming)
```
## define the feature  assume the k features from last are feature columns.
feature_cols = df_train.column[k::]
df_train_vec = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid='keep').transform(df_train)

N = 8  # number of worker
xgb = XGBoostClassifier( num_workers= N,
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        missing=0.0,
                        )
model = xgb.fit(df_train_vec)

```

(5) Obtain the result on test dataset

```
# predict on test dataset
df_test_vec = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid='keep').transform(df_test)

predict_df = model.transform(df_test_vec)
predict_df.show()
```

If we do fine tuning, here is the example:

```
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

xgbEval = BinaryClassificationEvaluator()

# Define your list of grid search parameters

paramGrid = (ParamGridBuilder()
             .addGrid(xgb.alpha,[1e-5, 1e-2, 0.1])
             .addGrid(xgb.eta, [0.001, 0.01])
             .addGrid(xgb.numRound, [150,160])
             .addGrid(xgb.maxDepth, range(3,7,3))
             .addGrid(xgb.minChildWeight, [3.0, 4.0])
             .addGrid(xgb.gamma, [i/10.0 for i in range(0,2)])
             .addGrid(xgb.colsampleBytree, [i/10.0 for i in range(3,6)])
             .addGrid(xgb.subsample, [0.4,0.6])
             .build())

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=xgbEval, numFolds=3)
cvModel = cv.fit(df_train)
cvPreds = cvModel.transform(df_test)
xgbEval.evaluate(cvPreds)

cvModel.bestModel.extractParamMap()
```

(6) Calculate the metrics

```
predictionAndLabels = predictions.select('prediction', F.col('label').cast(DoubleType())).rdd

metrics = MulticlassMetrics(predictionAndLabels)

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

```


##### Reference:

https://spark.apache.org/docs/2.2.0/mllib-evaluation-metrics.html

https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html#prepare-the-necessary-packages

https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/spark/estimator.py