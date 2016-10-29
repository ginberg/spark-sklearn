## Parallelized GridSearchCV in Spark with StratifiedShuffleSplit

I have run into an issue with using https://github.com/databricks/spark-sklearn with a StratifiedShuffleSplit cross validator. Therefore I have created this class.

### Example

from sklearn import svm, grid_search, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from spark_gridsearch import GridSearchCVSSS
iris = datasets.load_iris()

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svr = svm.SVC()
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.5)
clf = GridSearchCVSSS(sc, svr, parameters, cv=sss)
clf.fit(iris.data, iris.target)
