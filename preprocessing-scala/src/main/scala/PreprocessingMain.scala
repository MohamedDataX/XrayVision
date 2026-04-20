import java.io.File
import scala.util.Random
import org.apache.spark.sql.SparkSession

/**
 * preprocessing des images X-ray pour détection d'objets (17 classes)
 * RGB -> Grayscale
 * resize 256×256
 * conservation des bounding boxes (format YOLO)
 * split train/val/test
 */
object PreprocessingMain {
  
  def main(args: Array[String]): Unit = {
    println("""XRAYVISION
    PREPROCESSING (SPARK)
    """.stripMargin)
    
    val startTime = System.currentTimeMillis()
    Random.setSeed(Config.RANDOM_SEED)
    
    // init Spark
    println("Initialisation Apache Spark...")
    val spark = SparkSession.builder()
      .appName("XrayVision-Preprocessing")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "4g")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()
    
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    println(s"Spark initialisé avec ${sc.defaultParallelism} cœurs")
    
    //scanner les images ++
    println("\n[1/5] Scan des images positives...")
    val positiveSamples = DatasetScanner.scanPositiveSamples(new File(s"${Config.RAW_DATA_PATH}/Positive_Samples"))
    println(s"   ✓ ${positiveSamples.length} échantillons positifs trouvés")
    
    //scanner les images --
    println("\n[2/5] Scan des images négatives...")
    val negativeSamples = DatasetScanner.scanNegativeSamples(new File(s"${Config.RAW_DATA_PATH}/Negative_Samples"))
    println(s"${negativeSamples.length} échantillons négatifs trouvés")
    
    if (positiveSamples.isEmpty && negativeSamples.isEmpty) {
      println("❌ Aucune image trouvée!")
      spark.stop()
      return
    }
    
    //doss de sortie
    println("\n[3/5] Création structure de sortie...")
    createOutputDirs()

    //split data
    println("\n[4/5] Traitement des images avec Spark (grayscale + 256×256)...")
    val allSamples = Random.shuffle(positiveSamples ++ negativeSamples)
    val (trainSamples, valSamples, testSamples) = splitData(allSamples)
    
    SparkProcessing.processAndSave(trainSamples, "train", sc)
    SparkProcessing.processAndSave(valSamples, "val", sc)
    SparkProcessing.processAndSave(testSamples, "test", sc)
    
    //data.yaml
    println("\n[5/5] Création data.yaml...")
    YamlGenerator.create()
    
    spark.stop()
    
    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0

  }
  
  private def createOutputDirs(): Unit = {
    Seq("train", "val", "test").foreach { split =>
      new File(s"${Config.OUTPUT_PATH}/images/$split").mkdirs()
      new File(s"${Config.OUTPUT_PATH}/labels/$split").mkdirs()
    }
    println("Structure créée")
  }
  
  private def splitData(samples: Seq[Sample]): (Seq[Sample], Seq[Sample], Seq[Sample]) = {
    val trainCount = (samples.length * Config.TRAIN_RATIO).toInt
    val valCount = (samples.length * Config.VAL_RATIO).toInt
    
    val train = samples.take(trainCount)
    val valData = samples.slice(trainCount, trainCount + valCount)
    val test = samples.drop(trainCount + valCount)
    
    (train, valData, test)
  }
}
