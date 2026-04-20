package com.xrayvision.config

import java.nio.file.Paths


object Config {
  val _PROJECT_ROOT: String = Paths.get("").toAbsolutePath.getParent.getParent.toString
  val _FINETUNED_MODEL: String = s"${_PROJECT_ROOT}/training-python/models_fine_tuned/best_model.onnx"
  val _LOCAL_MODEL: String = s"${Paths.get("").toAbsolutePath.getParent}/models/best_model.onnx"

  val MODEL_PATH: String = "/Users/mohamedaitsidihou/Desktop/Projects/Xray/XrayVision/training-python/models_fine_tuned/best_model.onnx"

  val IMAGE_SIZE: Int = 256
  val NUM_CLASSES: Int = 18 // 17 objets + 1 background (index 0)

  // Index 0 = Background, Index 1-17 = Objets
  val CLASS_NAMES: Array[String] = Array(
    "Background",
    "Gun", "Knife", "Scissors", "Bullet", "Razor_blade", "Shuriken",
    "Lighter", "Pressure_vessel", "Wrench", "Pliers", "Hammer",
    "Screwdriver", "Battery", "Bat", "Saw_blade", "Fireworks", "Dart"
  )

  val DANGEROUS_CLASSES: Set[String] = Set("Gun", "Knife", "Bullet", "Razor_blade")

  val CONF_THRESHOLD: Float = 0.7f
  val NMS_THRESHOLD: Float = 0.3f
  val MAX_DETECTIONS: Int = 20

  // HTTP Server
  val HTTP_PORT: Int = 8000
  val HTTP_HOST: String = "0.0.0.0"

  // Spark
  val SPARK_APP_NAME: String = "XrayVision-InferenceService"
  val SPARK_MASTER: String = "local[*]"
  val SPARK_DRIVER_MEMORY: String = "4g"
}
