package com.xrayvision.core

import ai.onnxruntime.{OrtEnvironment, OrtSession, OnnxTensor}
import com.xrayvision.config.Config
import java.nio.FloatBuffer
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._


class OnnxModelWrapper(modelPath: String) {
  private val env: OrtEnvironment = OrtEnvironment.getEnvironment
  private val session: OrtSession = {
    if (Files.exists(Paths.get(modelPath))) {
      println(s"Modèle ONNX chargé: $modelPath")
      env.createSession(modelPath)
    } else {
      println(s"Modèle introuvable: $modelPath — inférence désactivée")
      null
    }
  }

  val isLoaded: Boolean = session != null
  val anchors: Array[Array[Float]] = InferenceHelpers.generate_anchors()

  def run(imagePixels: Array[Float]): Option[(Array[Array[Float]], Array[Array[Float]])] = {
    if (!isLoaded) return None
    val shape = Array(1L, 1L, Config.IMAGE_SIZE.toLong, Config.IMAGE_SIZE.toLong)
    val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imagePixels), shape)
    val inputs = Map(session.getInputNames.iterator.next() -> tensor).asJava
    val results = session.run(inputs)

    // ONNX retourne [[[F (batch x anchors x classes) — on extrait le batch 0
    val cls_raw = results.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
    val reg_raw = results.get(1).getValue.asInstanceOf[Array[Array[Array[Float]]]]

    val cls_preds = cls_raw(0) // shape: [num_anchors, NUM_CLASSES]
    val reg_preds = reg_raw(0) // shape: [num_anchors, 4]

    tensor.close()
    results.close()
    Some((cls_preds, reg_preds))
  }
}
