package com.xrayvision.models

/**
 * Détection pour une image — une annotation bounding box avec classe
 */
case class Detection(
  `class`: String,
  class_id: Int,
  confidence: Float,
  bbox: List[Float], // [x1, y1, x2, y2] normalized [0..1]
  bbox_pixels: List[Int],
  is_dangerous: Boolean
)

/**
 * Réponse HTTP pour POST /predict
 */
case class PredictResponse(
  filename: String,
  image_size: List[Int],
  detections: List[Detection],
  num_detections: Int,
  has_dangerous: Boolean,
  dangerous_items: List[String],
  summary: String
)

/**
 * Réponse HTTP pour GET /health
 */
case class HealthResponse(
  status: String,
  model_loaded: Boolean,
  num_classes: Int,
  classes: List[String]
)
