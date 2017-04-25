package ro.sandd


import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.log4j.{Level, LogManager}
import scopt.OptionParser
import org.apache.spark.mllib.evaluation.RegressionMetrics


/**
  * Hello world!
  *
  */
object App {
  val log = LogManager.getLogger("ro.sandd.App")
  log.setLevel(Level.INFO)

  val spark = SparkSession
    .builder()
    .appName("Data Science")
    .master("local[2]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  case class Params(trainInput: String = "", testInput: String = "",
                    outputFile: String = "",
                    algorithm: String = "",
                    algoMaxIter: Seq[Int] = Seq(30),
                    algoMaxDepth: Seq[Int] = Seq(3),
                    algoMaxBins: Seq[Int] = Seq(32),
                    algoNumTrees: Seq[Int] = Seq(3),
                    numFolds: Int = 10,
                    trainSample: Double = 1.0,
                    testSample: Double = 1.0)

  def main(args: Array[String]): Unit = {
    /*
     * Reading command line parameters
     */

    val parser = new OptionParser[Params]("DataScience") {
      head("DataScience", "1.0")

      opt[String]("trainInput").required().valueName("<file>").action((x, c) =>
        c.copy(trainInput = x)).text("Path to file/directory for training data")

      opt[String]("testInput").required().valueName("<file>").action((x, c) =>
        c.copy(testInput = x)).text("Path to file/directory for test data")

      opt[String]("outputFile").valueName("<file>").action((x, c) =>
        c.copy(outputFile = x)).text("Path to output file")

      opt[String]("algorithm").required().valueName("<algorithm>").action((x, c) =>
        c.copy(algorithm = x)).text("RF = RandomForest, GBT = Gradient Boosted Trees")

      opt[Seq[Int]]("algoNumTrees").valueName("<n1[,n2,n3...]>").action((x, c) =>
        c.copy(algoNumTrees = x)).text("One or more options for number of trees for RandomForest model. Default: 3")

      opt[Seq[Int]]("algoMaxIter").valueName("<n1[,n2,n3...]>").action((x, c) =>
        c.copy(algoMaxIter = x)).text("One or more values for limit of iterations. Default: 30")

      opt[Seq[Int]]("algoMaxDepth").valueName("<n1[,n2,n3...]>").action((x, c) =>
            c.copy(algoMaxDepth = x)).text("One or more values for depth limit. Default: 3")

      opt[Seq[Int]]("algoMaxBins").valueName("<n1[,n2,n3...]>").action((x, c) =>
        c.copy(algoMaxBins = x)).text("One or more values for depth limit. Default: 32")

      opt[Int]("numFolds").action((x, c) =>
        c.copy(numFolds = x)).text("Number of folds for K-fold Cross Validation. Default: 10")

      opt[Double]("trainSample").action((x, c) =>
        c.copy(trainSample = x)).text("Sample fraction from 0.0 to 1.0 for train data")

      opt[Double]("testSample").action((x, c) =>
        c.copy(testSample = x)).text("Sample fraction from 0.0 to 1.0 for test data")

    }

    parser.parse(args, Params()) match {
      case Some(params) =>
        if(params.algorithm == "GBT")
          processGBT(params)
        else
          processRandomForest(params);
      case None =>
        throw new IllegalArgumentException("One or more parameters are invalid or missing")
    }
  }

  def processGBT(params: Params): Unit = {
    // ******************************************************
    log.info(s"Reading data from ${params.trainInput} file")
    // ******************************************************
    val dfTrain = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("dateFormat", "dd.MM.yyyy HH:mm")
      .csv(params.trainInput)

    // *****************************************************
    log.info(s"Reading data from ${params.testInput} file")
    // *****************************************************
    val dfTest = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("dateFormat", "dd.MM.yyyy HH:mm")
      .csv(params.testInput)


    // *******************************************
    log.info("Preparing data for training model")
    // *******************************************

    // Preprocess data to fix, clean and build features
    val dfFeaturedTraining = preprocess(dfTrain).sample(false, params.trainSample)
    val testSet = preprocess(dfTest).sample(false, params.testSample).cache

    // Split training data
    val splits = dfFeaturedTraining.randomSplit(Array(0.7, 0.3))
    val (trainingSet, validationSet) = (splits(0), splits(1))

    trainingSet.cache()
    validationSet.cache()
    // **************************************************
    log.info("Building Machine Learning pipeline")
    // **************************************************

    // Index Categorical columns
    val stringIndexerStages = trainingSet.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(categNewCol(c))
        .fit(dfFeaturedTraining.select(c).union(testSet.select(c))))

    // Definitive set of feature columns
    val featureCols = trainingSet.columns
      .filter(onlyFeatureCols)
      .map(categNewCol)

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val algorhytm = new GBTRegressor().setLabelCol("label").setFeaturesCol("features")

    val pipeline = new Pipeline().setStages((stringIndexerStages :+ assembler) :+ algorhytm)

    // ***********************************************************
    log.info("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************

    val paramGrid = new ParamGridBuilder()
      .addGrid(algorhytm.maxIter, params.algoMaxIter)
      .addGrid(algorhytm.maxDepth, params.algoMaxDepth)
      .addGrid(algorhytm.maxBins, params.algoMaxBins)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(params.numFolds)

    // ************************************************************
    log.info("Training model with GradientBoostedTrees algorithm")
    // ************************************************************

    val cvModel = cv.fit(trainingSet)

    // **********************************************************************
    log.info("Evaluating model on train and test data and calculating RMSE")
    // **********************************************************************

    val trainPredictionsAndLabels = cvModel.transform(trainingSet).select("label", "prediction").map(row => (row.getInt(0).toDouble, row.getDouble(1))).rdd

    val validPredictionsAndLabels = cvModel.transform(validationSet).select("label", "prediction").map(row => (row.getInt(0).toDouble, row.getDouble(1))).rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val featureImportances = bestModel.stages.last.asInstanceOf[GBTRegressionModel].featureImportances.toArray

    val output = "\n=====================================================================\n" +
      s"Param trainSample: ${params.trainSample}\n" +
      s"Param testSample: ${params.testSample}\n" +
      s"TrainingData count: ${trainingSet.count}\n" +
      s"ValidationData count: ${validationSet.count}\n" +
      s"TestData count: ${testSet.count}\n" +
      "=====================================================================\n" +
      s"Param maxIter = ${params.algoMaxIter.mkString(",")}\n" +
      s"Param maxDepth = ${params.algoMaxDepth.mkString(",")}\n" +
      s"Param numFolds = ${params.numFolds}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"GBT features importances:\n ${featureCols.zip(featureImportances).map(t => s"\t${t._1} = ${t._2}").mkString("\n")}\n" +
      "=====================================================================\n"

    log.info(output)

    // *****************************************
    log.info("Run prediction over test dataset")
    // *****************************************
    cvModel.transform(testSet).select("label", "prediction").show()

    if(!params.outputFile.isEmpty){
      cvModel.transform(testSet)
        .select("label", "prediction")
        .coalesce(1)
        .write.format("csv")
        .option("header", "true")
        .save(params.outputFile)
    }
  }

  def processRandomForest(params: Params): Unit = {
    // ******************************************************
    log.info(s"Reading data from ${params.trainInput} file")
    // ******************************************************
    val dfTrain = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("dateFormat", "dd.MM.yyyy HH:mm")
      .csv(params.trainInput)

    // *****************************************************
    log.info(s"Reading data from ${params.testInput} file")
    // *****************************************************
    val dfTest = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("dateFormat", "dd.MM.yyyy HH:mm")
      .csv(params.testInput)


    // *******************************************
    log.info("Preparing data for training model")
    // *******************************************

    // Preprocess data to fix, clean and build features
    val dfFeaturedTraining = preprocess(dfTrain).sample(false, params.trainSample)
    val dfFeaturedTest = preprocess(dfTest).sample(false, params.testSample).cache

    // Split training data
    val splits = dfFeaturedTraining.randomSplit(Array(0.7, 0.3))
    val (trainingSet, validationSet) = (splits(0), splits(1))

    trainingSet.cache()
    validationSet.cache()
    // **************************************************
    log.info("Building Machine Learning pipeline")
    // **************************************************

    // Index Categorical columns
    val stringIndexerStages = trainingSet.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(categNewCol(c))
        .fit(dfFeaturedTraining.select(c).union(dfFeaturedTest.select(c))))

    // Definitive set of feature columns
    val generatedFeatureColumns = List("airline1", "fType1", "airport1", "acType1", "airline_airport1", "airline_actype1", "hour_airline1",
      "hour_actype1", "hour_intdom1")
    val allTableColumns = trainingSet.columns ++ generatedFeatureColumns
    val featureCols = allTableColumns
      .filter(onlyFeatureCols)
      .map(categNewCol)

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val algorhytm = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("label")

    val pipeline = new Pipeline().setStages((stringIndexerStages :+ assembler) :+ algorhytm)

    // ***********************************************************
    log.info("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************

    val paramGrid = new ParamGridBuilder()
      .addGrid(algorhytm.numTrees, params.algoNumTrees)
      .addGrid(algorhytm.maxDepth, params.algoMaxDepth)
      .addGrid(algorhytm.maxBins, params.algoMaxBins)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(params.numFolds)

    // ************************************************************
    log.info("Training model with RandomForest algorithm")
    // ************************************************************

    val cvModel = cv.fit(trainingSet)

    // **********************************************************************
    log.info("Evaluating model on train and test data and calculating RMSE")
    // **********************************************************************

    val trainPredictionsAndLabels = cvModel.transform(trainingSet).select("label", "prediction").map(row => (row.getInt(0).toDouble, row.getDouble(1))).rdd

    val validPredictionsAndLabels = cvModel.transform(validationSet).select("label", "prediction").map(row => (row.getInt(0).toDouble, row.getDouble(1))).rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val featureImportances = bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].featureImportances.toArray

    val output = "\n=====================================================================\n" +
      s"Param trainSample: ${params.trainSample}\n" +
      s"Param testSample: ${params.testSample}\n" +
      s"TrainingData count: ${trainingSet.count}\n" +
      s"ValidationData count: ${validationSet.count}\n" +
      s"TestData count: ${dfFeaturedTest.count}\n" +
      "=====================================================================\n" +
      s"Param algoNumTrees = ${params.algoNumTrees.mkString(",")}\n" +
      s"Param algoMaxDepth = ${params.algoMaxDepth.mkString(",")}\n" +
      s"Param algoMaxBins = ${params.algoMaxBins.mkString(",")}\n" +
      s"Param numFolds = ${params.numFolds}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"RandomForest features importances:\n ${featureCols.zip(featureImportances).map(t => s"\t${t._1} = ${t._2}").mkString("\n")}\n" +
      "=====================================================================\n"

    log.info(output)

    // *****************************************
    log.info("Run prediction over test dataset")
    // *****************************************
    cvModel.transform(dfFeaturedTest).select("label", "prediction").show()

    if(!params.outputFile.isEmpty){
      cvModel.transform(dfFeaturedTest)
        .select("label", "prediction")
        .coalesce(1)
        .write.format("csv")
        .option("header", "true")
        .save(params.outputFile)
    }
  }

  def preprocess(df: DataFrame) = {
    val fixedNamesDf = fixColumnNames(df)
    val cleanedDf = filterCorruptedData(fixedNamesDf)
    val featuredDf = fixDataAndBuildFeatures(cleanedDf)

    featuredDf
  }

  def fixColumnNames(df: DataFrame) = {
    df.withColumnRenamed("Sched date local", "sDate")
      .withColumnRenamed("Actual", "aDate")
      .withColumnRenamed("Flightnumber", "fNo")
      .withColumnRenamed("Airline", "airline")
      .withColumnRenamed("Arrdep", "arr_dep")
      .withColumnRenamed("Intdom5", "int_dom")
      .withColumnRenamed("Flt type", "fType")
      .withColumnRenamed("Airport1", "airport")
      .withColumnRenamed("Ac type", "acType")
      .withColumnRenamed("Seats avail", "seatsAvailable")
      .withColumnRenamed("Local", "label") // We will use this as the label column
      .withColumnRenamed("DomDom", "domdom")
      .withColumnRenamed("DomInt", "domint")
      .withColumnRenamed("IntDom13", "intdom")
      .withColumnRenamed("IntIntl", "intint")
      .withColumnRenamed("Total", "total")
  }

  def filterCorruptedData(df: DataFrame) = {
    df.filter($"acType".isNotNull)
      .filter($"seatsAvailable".isNotNull)
      .filter($"seatsAvailable" > $"local")
  }

  def fixDataAndBuildFeatures(df: DataFrame) = {
    df.withColumn("sDate", unix_timestamp($"sDate", "dd.MM.yyyy HH:mm").cast("timestamp"))
      .withColumn("aDate", unix_timestamp($"aDate", "dd.MM.yyyy HH:mm:ss").cast("timestamp"))
      .withColumn("is_arrival", when($"arr_dep" === "A", 1)
        .otherwise(0))
      .withColumn("isInt", when($"int_dom" === "I", 1)
        .otherwise(0))
      .withColumn("hour_of_day", hour($"sDate"))
      .withColumn("month", month($"sDate"))
      .withColumn("time_of_day", when($"hour_of_day".between(5, 11), 1)
        .when($"hour_of_day".between(12, 17), 2)
        .when($"hour_of_day".between(17, 24), 3)
        .otherwise(4)) // Dimineata/Pranz/Seara/Noaptea
      .withColumn("isDay", when($"hour_of_day".between(6, 24), 1)
      .otherwise(0))
      .withColumn("season", when($"month".between(1, 2), 1)
        .when($"month".between(3, 5), 2)
        .when($"month".between(6, 8), 3)
        .when($"month".between(9, 11), 4)
        .otherwise(1)) // Iarna/Primvara/Vara/Toamna
      .withColumn("delay", (($"aDate".cast("long") - $"sDate".cast("long")) / 60).cast("integer"))
      .withColumn("airline_airport", concat($"airline", $"airport"))
      .withColumn("airline_actype", concat($"airline", $"acType"))
      .withColumn("hour_airline", concat($"airline", $"hour_of_day"))
      .withColumn("hour_airline_airport", concat($"airline", $"hour_of_day", $"airport"))
      .withColumn("month_airline_airport", concat($"airline", $"month", $"airport"))
      .withColumn("season_airline_airport", concat($"airline", $"season", $"airport"))
      .withColumn("hour_actype", concat($"acType", $"hour_of_day"))
      .withColumn("hour_intdom", concat($"int_dom", $"hour_of_day"))
      .withColumn("airline_arrdep", concat($"airline", $"arr_dep"))
      .withColumn("actype_arrdep", concat($"acType", $"arr_dep"))
  }

  def onlyFeatureCols(c: String): Boolean = {
    val featureColumns = List("seatsAvailable", "local", "is_arrival", "isInt", "hour_of_day", "month", "time_of_day",
      "isDay", "season", "airline1", "fType1", "airport1", "acType1", "airline_actype1", "hour_intdom1", "hour_actype1", "hour_airline1", "airline_airport1")
    featureColumns.contains(c)
  }

  def isCateg(c: String): Boolean = {
    val columnsToBeIndexed = List("airline", "fType", "airport", "acType", "airline_airport", "airline_actype", "hour_airline",
      "hour_airline_airport", "month_airline_airport", "season_airline_airport", "hour_actype", "hour_intdom", "airline_arrdep", "actype_arrdep")
    columnsToBeIndexed.contains(c)
  }

  def categNewCol(c: String): String = if (isCateg(c)) c + "1" else c

  class Cleaner extends Transformer {
    override def transform(dataset: Dataset[_]): DataFrame = {
      dataset.select("local", "features").withColumn("label", $"local".cast("double")).drop("local")
    }

    override def copy(extra: ParamMap): Transformer = this

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = {
      val s = StructType(schema
        .filter(field => {
          field.name == "local" || field.name == "features"
        })
        .map(field => {
          if (field.name == "local") {
            field.copy(name = "label", dataType = DoubleType)
          } else {
            field
          }
        }))
      s
    }

    override val uid: String = "mmm"
  }

}
