package com.xrayvision.core

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import com.xrayvision.config.Config
import com.xrayvision.models.Detection


object InferenceHelpers {

  /**
   * preprocess_image — convertit en niveaux de gris, resize, normalise
   * Retourne un tableau Float[1, 1, IMAGE_SIZE, IMAGE_SIZE]
   */
  def preprocess_image(imageBytes: Array[Byte]): Array[Float] = {
    val img = ImageIO.read(new ByteArrayInputStream(imageBytes))
    val resized = new BufferedImage(Config.IMAGE_SIZE, Config.IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY)
    val g = resized.createGraphics()
    g.drawImage(img, 0, 0, Config.IMAGE_SIZE, Config.IMAGE_SIZE, null)
    g.dispose()

    val raster = resized.getData
    val pixels = new Array[Float](Config.IMAGE_SIZE * Config.IMAGE_SIZE)
    for (i <- pixels.indices) {
      val sample = new Array[Int](1)
      raster.getPixel(i % Config.IMAGE_SIZE, i / Config.IMAGE_SIZE, sample)
      pixels(i) = sample(0) / 255.0f
    }
    pixels // shape sera [1, 1, 256, 256] une fois enveloppé dans OnnxTensor
  }

  def decode_predictions(
    cls_preds: Array[Array[Float]],
    reg_preds: Array[Array[Float]],
    anchors: Array[Array[Float]],
    conf_threshold: Float = Config.CONF_THRESHOLD,
    nms_threshold: Float = Config.NMS_THRESHOLD
  ): List[Detection] = {

    val numAnchors = cls_preds.length

    // Softmax ligne par ligne — ignore classe 0 (Background)
    val objectProbs = cls_preds.map { logits =>
      val maxLogit = logits.max
      val exp = logits.map(l => math.exp(l - maxLogit).toFloat)
      val sumExp = exp.sum
      val probs = exp.map(_ / sumExp)
      probs.slice(1, Config.NUM_CLASSES) // Exclure index 0 (Background)
    }

    // scores et labels (avec offset +1 pour retrouver l'index réel)
    val scored = objectProbs.zipWithIndex.map { case (probs, anchorIdx) =>
      val labelOffset = probs.zipWithIndex.maxBy(_._1)
      val score = labelOffset._1
      val label = labelOffset._2 + 1 // remettre l'offset
      (score, label, anchorIdx)
    }

    val filtered = scored.filter(_._1 > conf_threshold)
    if (filtered.isEmpty) return List.empty

    // Decode boxes 
    val boxes = filtered.map { case (score, label, i) =>
      val anchor = anchors(i)
      val reg = reg_preds(i)
      val cx = anchor(0) + reg(0) * anchor(2)
      val cy = anchor(1) + reg(1) * anchor(3)
      val w = anchor(2) * math.exp(math.min(reg(2), 4.0)).toFloat
      val h = anchor(3) * math.exp(math.min(reg(3), 4.0)).toFloat
      val x1 = math.max(0f, cx - w / 2)
      val y1 = math.max(0f, cy - h / 2)
      val x2 = math.min(1f, cx + w / 2)
      val y2 = math.min(1f, cy + h / 2)
      (score, label, Array(x1, y1, x2, y2))
    }

    // Simple NMS
    val kept = simple_nms(
      boxes.map(_._3),
      boxes.map(_._1),
      nms_threshold
    ).take(Config.MAX_DETECTIONS)

    kept.map { idx =>
      val (score, labelId, box) = boxes(idx)
      val className = Config.CLASS_NAMES(labelId)
      Detection(
        `class` = className,
        class_id = labelId,
        confidence = math.round(score * 1000).toFloat / 1000f,
        bbox = box.map(x => math.round(x * 10000).toFloat / 10000f).toList,
        bbox_pixels = List.empty, // rempli dans /predict avec la taille réelle
        is_dangerous = Config.DANGEROUS_CLASSES.contains(className)
      )
    }.toList
  }

// simple NMS
  def simple_nms(
    boxes: Array[Array[Float]],
    scores: Array[Float],
    threshold: Float
  ): List[Int] = {
    if (boxes.isEmpty) return List.empty

    var order = scores.zipWithIndex.sortBy(-_._1).map(_._2).toList
    val keep = scala.collection.mutable.ListBuffer[Int]()

    while (order.nonEmpty) {
      val i = order.head
      keep += i
      order = order.tail

      order = order.filter { j =>
        val xx1 = math.max(boxes(i)(0), boxes(j)(0))
        val yy1 = math.max(boxes(i)(1), boxes(j)(1))
        val xx2 = math.min(boxes(i)(2), boxes(j)(2))
        val yy2 = math.min(boxes(i)(3), boxes(j)(3))

        val w = math.max(0f, xx2 - xx1)
        val h = math.max(0f, yy2 - yy1)
        val inter = w * h

        val area_i = (boxes(i)(2) - boxes(i)(0)) * (boxes(i)(3) - boxes(i)(1))
        val area_j = (boxes(j)(2) - boxes(j)(0)) * (boxes(j)(3) - boxes(j)(1))
        val union = area_i + area_j - inter + 1e-6f
        val iou = inter / union

        iou < threshold
      }
    }
    keep.toList
  }


  def generate_anchors(): Array[Array[Float]] = {
    val featureSizes = Array(32, 16, 8, 4)
    val anchorScales = Array(
      Array(0.1f, 0.15f, 0.2f),
      Array(0.2f, 0.3f, 0.4f),
      Array(0.4f, 0.5f, 0.6f),
      Array(0.6f, 0.7f, 0.8f)
    )
    val anchors = scala.collection.mutable.ArrayBuffer[Array[Float]]()
    for ((fmSize, scales) <- featureSizes.zip(anchorScales))
      for (i <- 0 until fmSize; j <- 0 until fmSize; scale <- scales) {
        val cx = (j + 0.5f) / fmSize
        val cy = (i + 0.5f) / fmSize
        anchors += Array(cx, cy, scale, scale)
      }
    anchors.toArray
  }
}
