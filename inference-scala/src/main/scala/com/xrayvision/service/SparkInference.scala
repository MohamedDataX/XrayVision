package com.xrayvision.service

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import com.xrayvision.core.{OnnxModelWrapper, InferenceHelpers}

/**
 * SparkInference — enregistrement de l'UDF Spark pour inférence distribuée
 */
object SparkInference {

  def registerUDF(spark: SparkSession, modelPath: String): Unit = {

    val inferUDF = udf((imageBytes: Array[Byte]) => {
      // Instanciation locale sur le worker pour qu'on évite la sérialisation de la session ONNX .. 
      val wrapper = new OnnxModelWrapper(modelPathBC.value)
      if (!wrapper.isLoaded || imageBytes == null) {
        Array.empty[Map[String, String]]
      } else {
        val pixels = InferenceHelpers.preprocess_image(imageBytes)
        wrapper.run(pixels) match {
          case None => Array.empty[Map[String, String]]
          case Some((cls_preds, reg_preds)) =>
            val detections = InferenceHelpers.decode_predictions(cls_preds, reg_preds, wrapper.anchors)
            detections.map(d => Map(
              "class" -> d.`class`,
              "class_id" -> d.class_id.toString,
              "confidence" -> d.confidence.toString,
              "is_dangerous" -> d.is_dangerous.toString,
              "bbox" -> d.bbox.mkString(",")
            )).toArray
        }
      }
    })

    spark.udf.register("xray_detect", inferUDF)
    println("UDF 'xray_detect' enregistrée dans SparkSession")
  }

   //runBatch — applique l'inférence sur un DataFrame contenant une colonne : image_bytes
  def runBatch(spark: SparkSession, df: DataFrame): DataFrame = {
    df.withColumn("detections", callUDF("xray_detect", col("image_bytes")))
  }
}
