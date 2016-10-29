## Parallelized GridSearchCV in Apache Spark with StratifiedShuffleSplit

I have run into an issue with using https://github.com/databricks/spark-sklearn with a StratifiedShuffleSplit cross validator. Therefore I have created this class.


### Use case

It focuses on problems that have a small amount of data and that can be run in parallel.
- for small datasets, it distributes the search for estimator parameters (`GridSearchCV` in scikit-learn), using Spark,
- for datasets that do not fit in memory, I recommend using the [distributed implementation in Spark ML](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html).
- StratifiedShuffleSplit is used as cross validator, this makes sure that every fold preserves the percentage of samples for each class and these folds are randomized. 

### Example

```python
from sklearn import svm, grid_search, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from spark_gridsearch import GridSearchCVSSS
iris = datasets.load_iris()

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svr = svm.SVC()
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.5)

# first argument (sc) is the sparkContext, this should be available. You might need to import it yourself. 
# I used it with jupyter notebook and pyspark where I don't need to import it in the notebook itself.
clf = GridSearchCVSSS(sc, svr, parameters, cv=sss)
clf.fit(iris.data, iris.target)
```

## License

This package is released under the Apache 2.0 license.
