import org.apache.spark.ml.classification.{LogisticRegression,DecisionTreeClassifier,RandomForestClassificationModel,RandomForestClassifier}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Reading the turnover csv file.
var data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("turnover.csv")

// // Printing the Schema of the DataFrame
// data.printSchema()
//
// data.filter($"satisfaction_level".isNull ||
// $"last_evaluation".isNull ||
// $"number_project".isNull ||
// $"average_montly_hours".isNull ||
// $"time_spend_company".isNull ||
// $"Work_accident".isNull ||
// $"left".isNull ||
// $"promotion_last_5years".isNull ||
// $"sales".isNull ||
// $"salary".isNull).count()


// Renaming columns for more readability
val colNames = Seq("satisfaction", "evaluation", "projectCount", "averageMonthlyHours", "yearsAtCompany",
                  "workAccident", "turnover", "promotion", "department", "salary")
val renamedData = data.toDF(colNames: _*)
renamedData.printSchema()


///////////////////////////
// FEATURES SELECTION  ////
//////////////////////////
val featureseldata = (renamedData.select(renamedData("turnover").as("label"),
                    $"satisfaction",$"evaluation",$"projectCount", $"averageMonthlyHours",$"yearsAtCompany",
                    $"workAccident", $"promotion", $"department", $"salary"))

//Converting strings into numerical values
val departmentIndexer = new StringIndexer().setInputCol("department").setOutputCol("departmentIndex")
val salaryIndexer = new StringIndexer().setInputCol("salary").setOutputCol("salaryIndex")

//Converting numerical values into One Hot Encoding 0 or 1
val departmentEncoder = new OneHotEncoder().setInputCol("departmentIndex").setOutputCol("departmentVec")
val salaryEncoder = new OneHotEncoder().setInputCol("salaryIndex").setOutputCol("salaryVec")

// Creating a new VectorAssembler object called assembler for the feature
// columns as the input and setting the output column to be called features
val assembler = (new VectorAssembler()
                  .setInputCols(Array("satisfaction", "evaluation", "projectCount", "averageMonthlyHours",
                  "yearsAtCompany", "workAccident", "promotion", "departmentVec", "salaryVec"))
                  .setOutputCol("features"))

// Using randomSplit to create a train test split of 70/30
val Array(training, test) = featureseldata.randomSplit(Array(0.7, 0.3), seed = 12345)

// Setting Up the Pipeline
val rf = new RandomForestClassifier()

// CreatING a new pipeline with the previous stages
val pipeline = new Pipeline().setStages(Array(departmentIndexer,salaryIndexer,departmentEncoder,salaryEncoder,assembler,rf))

// Fitting the pipeline to training set.
val model = pipeline.fit(training)
val rfModel = model.stages(5).asInstanceOf[RandomForestClassificationModel].featureImportances.toArray
.take(10)
.foreach(println)


///////////////////////////////
//CLASSIFICATION MODELS //////
/////////////////////////////

//Creating common dataframe for all the classification models
val classdata = (renamedData.select(renamedData("turnover").as("label"),
                    $"satisfaction", $"projectCount", $"yearsAtCompany"))


// Creating a common VectorAssembler object called assembler for the feature
// columns as the input and setting the output column to be called features
val assembler = (new VectorAssembler()
                  .setInputCols(Array("satisfaction", "projectCount", "yearsAtCompany"))
                  .setOutputCol("features") )

// Using randomSplit to create common train test split of 70/30
val Array(training, test) = classdata.randomSplit(Array(0.7, 0.3))

///////////////////////////////
// LOGISTIC REGRESSION ///////
/////////////////////////////
// Creating a new LogisticRegression object called lr
val lr = new LogisticRegression()

// Creating a new pipeline with the stages: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// Fitting the pipeline to training set.
val model = pipeline.fit(training)

// Getting Results on Test Set with transform
val results = model.transform(test)

// ////////////////////////////////////
// //// MODEL EVALUATION /////////////
// //////////////////////////////////

// Convert the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiating a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Printing out the Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


///////////////////////////////
// DECISION TREE ///////
/////////////////////////////
// Creating a new DecisionTree object called dt
val dt = new DecisionTreeClassifier()

// Creating a new pipeline with the stages: assembler, dt

val pipeline = new Pipeline().setStages(Array(assembler, dt))

// Fitting the pipeline to training set.
val model = pipeline.fit(training)

// Getting Results on Test Set with transform
val results = model.transform(test)

// ////////////////////////////////////
// //// MODEL EVALUATION /////////////
// //////////////////////////////////

// Converting the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiating a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Printing out the Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

///////////////////////////////
// RANDOM FOREST /////////////
/////////////////////////////

// Creating a new RandomForest object called rf
val rf = new RandomForestClassifier()

// Creating a new pipeline with the stages: assembler, rf
val pipeline = new Pipeline().setStages(Array(assembler, rf))

// Fitting the pipeline to training set.
val model = pipeline.fit(training)

// Getting Results on Test Set with transform
val results = model.transform(test)

// ////////////////////////////////////
// //// MODEL EVALUATION /////////////
// //////////////////////////////////

// Converting the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiating a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Printing out the Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
