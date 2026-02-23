import java.io.File
import javax.imageio.ImageIO
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.commons.io.FileUtils

/**
 * Pipeline de traitement parallèle avec Spark
 */
object SparkProcessing {
  
  /** Traiter et sauvegarder les images avec Spark RDD */
  def processAndSave(samples: Seq[Sample], split: String, sc: SparkContext): Unit = {
    if (samples.isEmpty) {
      println(s"$split: 0 images (aucun échantillon)")
      return
    }
    
    val imageSize = Config.IMAGE_SIZE
    val outputPath = Config.OUTPUT_PATH
    val numPartitions = math.min(samples.length, sc.defaultParallelism * 2)
    
    val samplesRDD: RDD[Sample] = sc.parallelize(samples, numPartitions)
    
    val processedCount = samplesRDD.map { sample =>
      processSample(sample, split, imageSize, outputPath)
    }.reduce(_ + _)
    
    println(s"   ✓ $split: $processedCount/${samples.length} images traitées (Spark)")
  }
  
  private def processSample(sample: Sample, split: String, imageSize: Int, outputPath: String): Int = {
    try {
      val imageFile = new File(sample.imagePath)
      val original = ImageIO.read(imageFile)
      
      if (original == null) return 0
      
      val origWidth = original.getWidth
      val origHeight = original.getHeight
      
      // Traiter l'image
      val processed = ImageProcessing.process(original, imageSize)
      
      // Sauvegarder
      val baseName = imageFile.getName.replaceAll("\\.[^.]+$", "").replace("_OL", "").replace("_SD", "")
      ImageIO.write(processed, "png", new File(s"$outputPath/images/$split/$baseName.png"))
      
      // Créer labels YOLO
      val yoloLabels = sample.objects.map(obj => ImageProcessing.bboxToYolo(obj, origWidth, origHeight)).mkString("\n")
      FileUtils.writeStringToFile(new File(s"$outputPath/labels/$split/$baseName.txt"), yoloLabels, "UTF-8")
      
      1
    } catch {
      case _: Exception => 0
    }
  }
}
