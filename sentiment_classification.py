from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import sys
from functools import reduce

# Function for combining multiple DataFrames row-wise
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

if __name__ == "__main__":
    # Create a SparkContext and an SQLContext
    sc = SparkContext(appName="Sentiment Classification")
    sqlContext = SQLContext(sc)
    sc.setLogLevel('ERROR')
    # Load data
    # wholeTextFiles(path, [...]) reads a directory of text files from a filesystem
    # Each file is read as a single record and returned in a key-value pair
    # The key is the path and the value is the content of each file
    reviews = sc.wholeTextFiles('hdfs:///user/maria_dev/'+sys.argv[1]+'/*/*')

    # Create tuples: (class label, review text) - we ignore the file path
    # 1.0 for positive reviews
    # 0.0 for negative reviews
    reviews_f = reviews.map(lambda row: (1.0 if 'pos' in row[0] else 0.0, row[1]))

    # Convert data into a Spark SQL DataFrame
    # The first column contains the class label
    # The second column contains the review text
    dataset = reviews_f.toDF(['class_label', 'review'])

    # ----- PART II: FEATURE ENGINEERING -----

    # Tokenize the review text column into a list of words
    tokenizer = Tokenizer(inputCol='review', outputCol='words')
    words_data = tokenizer.transform(dataset)

    # Randomly split data into a training set, a development set and a test set
    # train = 60% of the data, dev = 20% of the data, test = 20% of the data
    # The random seed should be set to 42
    (train, dev, test) = words_data.randomSplit([.6, .2, .2], seed = 42)

    # Count the number of instances in, respectively, train, dev and test
    # Print the counts to standard output
    print('#train: ' + str(train.count()) + ', #dev: ' + str(dev.count()) +', #test: ' + str(test.count())) 

    # Count the number of positive/negative instances in, respectively, train, dev and test
    # Print the class distribution for each to standard output
    # The class distribution should be specified as the % of positive examples
    neg_train = train.groupBy("class_label").count().first()[1]
    neg_dev = dev.groupBy("class_label").count().first()[1]
    neg_test = test.groupBy("class_label").count().first()[1]
    print('% of positive examples train: ' + str((train.count()-neg_train)*100/train.count()) + '%')
    print('% of positive examples dev: ' + str((dev.count()-neg_dev)*100/dev.count()) + '%') 
    print('% of positive examples test: ' + str((test.count()-neg_test)*100/test.count()) + '%') 

    # Create a stopword list containing the 100 most frequent tokens in the training data
    # Hint: see below for how to convert a list of (word, frequency) tuples to a list of words
    words_train = train.select("words").rdd.flatMap(lambda p: p.words)
    freq_words = words_train.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    list_top100_tokens = sorted(freq_words.collect(), key=lambda x: x[1], reverse=True)[:100]
    stopwords = [frequency_tuple[0] for frequency_tuple in list_top100_tokens]

    # Replace the [] in the stopWords parameter with the name of your created list
    remover = StopWordsRemover(inputCol='words', outputCol='words_filtered', stopWords=stopwords)

    # Remove stopwords from all three subsets
    train_filtered = remover.transform(train)
    dev_filtered = remover.transform(dev)
    test_filtered = remover.transform(test)

    # Transform data to a bag of words representation
    # Only include tokens that have a minimum document frequency of 2
    cv = CountVectorizer(inputCol='words_filtered', outputCol='BoW', minDF=2.0)
    cv_model = cv.fit(train_filtered)
    train_data = cv_model.transform(train_filtered)
    dev_data = cv_model.transform(dev_filtered)
    test_data = cv_model.transform(test_filtered)
    
    # Print the vocabulary size (to STDOUT) after filtering out stopwords and very rare tokens
    # Hint: Look at the parameters of CountVectorizer
    print('Vocabulary size after filtering out stopwords and very rare tokens: '+ str(len(cv_model.vocabulary)))

    # Create a TF-IDF representation of the data
    idf = IDF(inputCol='BoW', outputCol='TFIDF')
    idf_model = idf.fit(train_data)
    train_tfidf = idf_model.transform(train_data)
    dev_tfidf = idf_model.transform(dev_data)
    test_tfidf = idf_model.transform(test_data)
    print(train_tfidf.count())

    # ----- PART III: MODEL SELECTION -----

    # Provide information about class labels: needed for model fitting
    # Only needs to be defined once for all models (but included in all pipelines)
    label_indexer = StringIndexer(inputCol = 'class_label', outputCol = 'label')

    # Create an evaluator for binary classification
    # Only needs to be created once, can be reused for all evaluation
    evaluator = BinaryClassificationEvaluator()

    # Train a decision tree with default parameters (including maxDepth=5)
    dt_classifier_default = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=5)

    # Create an ML pipeline for the decision tree model
    dt_pipeline_default = Pipeline(stages=[label_indexer, dt_classifier_default])

    # Apply pipeline and train model
    dt_model_default = dt_pipeline_default.fit(train_tfidf)

    # Apply model on devlopment data
    dt_predictions_default_dev = dt_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_dt_default_dev = evaluator.evaluate(dt_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev))

    # Check for signs of overfitting (by evaluating the model on the training set)
    dt_predictions_default_train = dt_model_default.transform(train_tfidf)
    auc_dt_default_train = evaluator.evaluate(dt_predictions_default_train, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train))

    # Tune the decision tree model by changing one of its hyperparameters
    # Build and evalute decision trees with the following maxDepth values: 3 and 4.
    dt_classifier_maxDepth3 = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=3)
    dt_classifier_maxDepth4 = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=4)
    dt_pipeline_maxDepth3 = Pipeline(stages=[label_indexer, dt_classifier_maxDepth3])
    dt_pipeline_maxDepth4 = Pipeline(stages=[label_indexer, dt_classifier_maxDepth4])
    dt_model_maxDepth3 = dt_pipeline_maxDepth3.fit(train_tfidf)
    dt_model_maxDepth4 = dt_pipeline_maxDepth4.fit(train_tfidf)
    dt_predictions_maxDepth3_dev = dt_model_maxDepth3.transform(dev_tfidf)
    dt_predictions_maxDepth4_dev = dt_model_maxDepth4.transform(dev_tfidf)
    auc_dt_maxDepth3_dev = evaluator.evaluate(dt_predictions_maxDepth3_dev, {evaluator.metricName: 'areaUnderROC'})
    auc_dt_maxDepth4_dev = evaluator.evaluate(dt_predictions_maxDepth4_dev, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, maxDepth = 3, Development Set, AUC: ' + str(auc_dt_maxDepth3_dev))
    print('Decision Tree, maxDepth = 4, Development Set, AUC: ' + str(auc_dt_maxDepth4_dev))
    
    # Train a random forest with default parameters (including numTrees=20)
    rf_classifier_default = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=20)

    # Create an ML pipeline for the random forest model
    rf_pipeline_default = Pipeline(stages=[label_indexer, rf_classifier_default])

    # Apply pipeline and train model
    rf_model_default = rf_pipeline_default.fit(train_tfidf)

    # Apply model on development data
    rf_predictions_default_dev = rf_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_rf_default_dev = evaluator.evaluate(rf_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Random Forest, Default Parameters, Development Set, AUC: ' + str(auc_rf_default_dev))

    # Check for signs of overfitting (by evaluating the model on the training set)
    rf_predictions_default_train = rf_model_default.transform(train_tfidf)
    auc_rf_default_train = evaluator.evaluate(rf_predictions_default_train, {evaluator.metricName: 'areaUnderROC'})
    print('Random Forest, Default Parameters, Training Set, AUC: ' + str(auc_rf_default_train))

    # Tune the random forest model by changing one of its hyperparameters
    # Build and evalute (on the dev set) another random forest with the following numTrees value: 100.
    rf_classifier_numTrees100 = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=100)
    rf_pipeline_numTrees100 = Pipeline(stages=[label_indexer, rf_classifier_numTrees100])
    rf_model_numTrees100 = rf_pipeline_numTrees100.fit(train_tfidf)
    rf_predictions_numTrees100_dev = rf_model_numTrees100.transform(dev_tfidf)
    auc_rf_numTrees100_dev = evaluator.evaluate(rf_predictions_numTrees100_dev, {evaluator.metricName: 'areaUnderROC'})
    print('Random Forest, numTrees = 100, Development Set, AUC: ' + str(auc_rf_numTrees100_dev))

    # ----- PART IV: MODEL EVALUATION -----

    # Create a new dataset combining the train and dev sets
    traindev_tfidf = unionAll(train_tfidf, dev_tfidf)

    # Evalute the best model on the test set
    # Build a new model from the concatenation of the train and dev sets in order to better utilize the data
    best_model_numTrees100 = rf_pipeline_numTrees100.fit(traindev_tfidf)
    best_predictions_numTrees100_test = best_model_numTrees100.transform(test_tfidf)
    auc_best_numTrees100_test = evaluator.evaluate(best_predictions_numTrees100_test, {evaluator.metricName: 'areaUnderROC'}) 
    print('Random Forest, numTrees = 100, Test Set, AUC: ' + str(auc_best_numTrees100_test))

