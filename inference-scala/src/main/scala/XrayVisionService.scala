/**
 * =============================================================================
 * XRAYVISION - Inference Service (Spark + Scala)
 * =============================================================================
 * Service d'inférence distribué pour détection d'objets dans les images X-ray
 * Traduit depuis main.py (FastAPI + PyTorch) vers Spark + Scala + ONNX Runtime
 * - POST /predict : retourne les détections avec bboxes
 * - GET  /health  : vérification du service
 */

import ai.onnxruntime.{OrtEnvironment, OrtSession, OnnxTensor}
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.http4s._
import org.http4s.dsl.io._
import org.http4s.implicits._
import org.http4s.blaze.server.BlazeServerBuilder
import org.http4s.circe._
import io.circe.syntax._
import io.circe.generic.auto._
import cats.effect.{IO, IOApp, Resource}
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, File}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters._
import java.nio.FloatBuffer
import java.nio.file.{Files, Paths}

// ============================================================================
// CONFIGURATION — mêmes noms que main.py
// ============================================================================
object Config {
  val _PROJECT_ROOT: String = Paths.get("").toAbsolutePath.getParent.getParent.toString
  val _FINETUNED_MODEL: String = s"${_PROJECT_ROOT}/training-python/models_fine_tuned/best_model.onnx"
  val _LOCAL_MODEL: String     = s"${Paths.get("").toAbsolutePath.getParent}/models/best_model.onnx"

  val MODEL_PATH: String = "/Users/mohamedaitsidihou/Desktop/Projects/Xray/XrayVision/training-python/models_fine_tuned/best_model.onnx"


  val IMAGE_SIZE: Int   = 256
  val NUM_CLASSES: Int  = 18   // 17 objets + 1 background (index 0)

  // Index 0 = Background, Index 1-17 = Objets
  val CLASS_NAMES: Array[String] = Array(
    "Background",
    "Gun", "Knife", "Scissors", "Bullet", "Razor_blade", "Shuriken",
    "Lighter", "Pressure_vessel", "Wrench", "Pliers", "Hammer",
    "Screwdriver", "Battery", "Bat", "Saw_blade", "Fireworks", "Dart"
  )

  val DANGEROUS_CLASSES: Set[String] = Set("Gun", "Knife", "Bullet", "Razor_blade")

  val CONF_THRESHOLD: Float = 0.7f
  val NMS_THRESHOLD: Float  = 0.3f
  val MAX_DETECTIONS: Int   = 20
}

// ============================================================================
// DOMAIN MODELS
// ============================================================================
case class Detection(
  `class`:     String,
  class_id:    Int,
  confidence:  Float,
  bbox:        List[Float],       // [x1, y1, x2, y2] normalized
  bbox_pixels: List[Int],
  is_dangerous: Boolean
)

case class PredictResponse(
  filename:        String,
  image_size:      List[Int],
  detections:      List[Detection],
  num_detections:  Int,
  has_dangerous:   Boolean,
  dangerous_items: List[String],
  summary:         String
)

case class HealthResponse(
  status:       String,
  model_loaded: Boolean,
  num_classes:  Int,
  classes:      List[String]
)

// ============================================================================
// INFERENCE HELPERS — même logique que main.py
// ============================================================================
object InferenceHelpers {

  /**
   * preprocess_image — convertit en niveaux de gris, resize, normalise
   * Retourne un tableau Float[1, 1, IMAGE_SIZE, IMAGE_SIZE]
   */
  def preprocess_image(imageBytes: Array[Byte]): Array[Float] = {
    val img      = ImageIO.read(new ByteArrayInputStream(imageBytes))
    val resized  = new BufferedImage(Config.IMAGE_SIZE, Config.IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY)
    val g        = resized.createGraphics()
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

  /**
   * decode_predictions — même logique de décodage SSD que main.py
   * cls_preds shape : [num_anchors, NUM_CLASSES]
   * reg_preds shape : [num_anchors, 4]
   * anchors   shape : [num_anchors, 4]  (cx, cy, w, h)
   */
  def decode_predictions(
    cls_preds:       Array[Array[Float]],
    reg_preds:       Array[Array[Float]],
    anchors:         Array[Array[Float]],
    conf_threshold:  Float = Config.CONF_THRESHOLD,
    nms_threshold:   Float = Config.NMS_THRESHOLD
  ): List[Detection] = {

    val numAnchors = cls_preds.length

    // Softmax ligne par ligne — ignore classe 0 (Background)
    val objectProbs = cls_preds.map { logits =>
      val maxLogit = logits.max
      val exp      = logits.map(l => math.exp(l - maxLogit).toFloat)
      val sumExp   = exp.sum
      val probs    = exp.map(_ / sumExp)
      probs.slice(1, Config.NUM_CLASSES)  // Exclure index 0 (Background)
    }

    // scores et labels (avec offset +1 pour retrouver l'index réel)
    val scored = objectProbs.zipWithIndex.map { case (probs, anchorIdx) =>
      val labelOffset = probs.zipWithIndex.maxBy(_._1)
      val score       = labelOffset._1
      val label       = labelOffset._2 + 1  // remettre l'offset
      (score, label, anchorIdx)
    }

    val filtered = scored.filter(_._1 > conf_threshold)
    if (filtered.isEmpty) return List.empty

    // Decode boxes (même formule que PyTorch)
    val boxes = filtered.map { case (score, label, i) =>
      val anchor = anchors(i)
      val reg    = reg_preds(i)
      val cx = anchor(0) + reg(0) * anchor(2)
      val cy = anchor(1) + reg(1) * anchor(3)
      val w  = anchor(2) * math.exp(math.min(reg(2), 4.0)).toFloat
      val h  = anchor(3) * math.exp(math.min(reg(3), 4.0)).toFloat
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
        `class`      = className,
        class_id     = labelId,
        confidence   = math.round(score * 1000).toFloat / 1000f,
        bbox         = box.map(x => math.round(x * 10000).toFloat / 10000f).toList,
        bbox_pixels  = List.empty,  // rempli dans /predict avec la taille réelle
        is_dangerous = Config.DANGEROUS_CLASSES.contains(className)
      )
    }.toList
  }

  /**
   * simple_nms — Non-Maximum Suppression identique à main.py
   */
  def simple_nms(
    boxes:     Array[Array[Float]],
    scores:    Array[Float],
    threshold: Float
  ): List[Int] = {
    if (boxes.isEmpty) return List.empty

    var order = scores.zipWithIndex.sortBy(-_._1).map(_._2).toList
    val keep  = scala.collection.mutable.ListBuffer[Int]()

    while (order.nonEmpty) {
      val i = order.head
      keep += i
      order = order.tail

      order = order.filter { j =>
        val xx1 = math.max(boxes(i)(0), boxes(j)(0))
        val yy1 = math.max(boxes(i)(1), boxes(j)(1))
        val xx2 = math.min(boxes(i)(2), boxes(j)(2))
        val yy2 = math.min(boxes(i)(3), boxes(j)(3))

        val w     = math.max(0f, xx2 - xx1)
        val h     = math.max(0f, yy2 - yy1)
        val inter = w * h

        val area_i = (boxes(i)(2) - boxes(i)(0)) * (boxes(i)(3) - boxes(i)(1))
        val area_j = (boxes(j)(2) - boxes(j)(0)) * (boxes(j)(3) - boxes(j)(1))
        val union  = area_i + area_j - inter + 1e-6f
        val iou    = inter / union

        iou < threshold
      }
    }
    keep.toList
  }

  /**
   * generate_anchors — même grille que SSDCNN256._generate_anchors (Python)
   */
  def generate_anchors(): Array[Array[Float]] = {
    val featureSizes   = Array(32, 16, 8, 4)
    val anchorScales   = Array(
      Array(0.1f, 0.15f, 0.2f),
      Array(0.2f, 0.3f,  0.4f),
      Array(0.4f, 0.5f,  0.6f),
      Array(0.6f, 0.7f,  0.8f)
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

// ============================================================================
// ONNX MODEL WRAPPER
// ============================================================================
class OnnxModelWrapper(modelPath: String) {
  private val env: OrtEnvironment = OrtEnvironment.getEnvironment
  private val session: OrtSession = {
    if (Files.exists(Paths.get(modelPath))) {
      println(s"✅ Modèle ONNX chargé: $modelPath")
      env.createSession(modelPath)
    } else {
      println(s"⚠️  Modèle introuvable: $modelPath — inférence désactivée")
      null
    }
  }

  val isLoaded: Boolean = session != null
  val anchors: Array[Array[Float]] = InferenceHelpers.generate_anchors()

  /**
   * run — exécute le modèle sur un batch d'une image
   * Retourne (cls_preds, reg_preds) de shape [num_anchors, NUM_CLASSES] / [num_anchors, 4]
   */

  def run(imagePixels: Array[Float]): Option[(Array[Array[Float]], Array[Array[Float]])] = {
   if (!isLoaded) return None
   val shape   = Array(1L, 1L, Config.IMAGE_SIZE.toLong, Config.IMAGE_SIZE.toLong)
   val tensor  = OnnxTensor.createTensor(env, FloatBuffer.wrap(imagePixels), shape)
   val inputs  = Map(session.getInputNames.iterator.next() -> tensor).asJava
   val results = session.run(inputs)

  // ONNX retourne [[[F (batch x anchors x classes) — on extrait le batch 0
   val cls_raw = results.get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]
   val reg_raw = results.get(1).getValue.asInstanceOf[Array[Array[Array[Float]]]]

   val cls_preds = cls_raw(0)  // shape: [num_anchors, NUM_CLASSES]
   val reg_preds = reg_raw(0)  // shape: [num_anchors, 4]

   tensor.close()
   results.close()
   Some((cls_preds, reg_preds))
 }
} 

// ============================================================================
// SPARK UDF — inférence distribuée sur un DataFrame d'images
// ============================================================================
object SparkInference {

  /**
   * Crée une UDF Spark qui applique l'inférence sur chaque ligne.
   * Chaque ligne contient les bytes bruts d'une image (BinaryType).
   * Retourne un tableau de maps (détections sérialisées).
   */
  def registerUDF(spark: SparkSession, modelPath: String): Unit = {
    // Broadcast du chemin pour éviter la sérialisation du wrapper
    val modelPathBC = spark.sparkContext.broadcast(modelPath)

    val inferUDF = udf((imageBytes: Array[Byte]) => {
      // Instanciation locale sur le worker (évite la sérialisation de la session ONNX)
      val wrapper = new OnnxModelWrapper(modelPathBC.value)
      if (!wrapper.isLoaded || imageBytes == null) Array.empty[Map[String, String]]
      else {
        val pixels = InferenceHelpers.preprocess_image(imageBytes)
        wrapper.run(pixels) match {
          case None => Array.empty[Map[String, String]]
          case Some((cls_preds, reg_preds)) =>
            val detections = InferenceHelpers.decode_predictions(cls_preds, reg_preds, wrapper.anchors)
            detections.map(d => Map(
              "class"        -> d.`class`,
              "class_id"     -> d.class_id.toString,
              "confidence"   -> d.confidence.toString,
              "is_dangerous" -> d.is_dangerous.toString,
              "bbox"         -> d.bbox.mkString(",")
            )).toArray
        }
      }
    })

    spark.udf.register("xray_detect", inferUDF)
    println("✅ UDF 'xray_detect' enregistrée dans SparkSession")
  }

  /**
   * runBatch — applique l'inférence sur un DataFrame contenant une colonne "image_bytes"
   */
  def runBatch(spark: SparkSession, df: DataFrame): DataFrame = {
    df.withColumn("detections", callUDF("xray_detect", col("image_bytes")))
  }
}

// ============================================================================
// HTTP SERVICE (http4s) — endpoints /predict et /health
// ============================================================================
object XrayVisionService extends IOApp.Simple {

  // Modèle chargé une seule fois (singleton sur le driver)
  private val model = new OnnxModelWrapper(Config.MODEL_PATH)

  // -------------------------------------------------------------------------
  // Routes
  // -------------------------------------------------------------------------
  val routes = HttpRoutes.of[IO] {

    // GET /health
    case GET -> Root / "health" =>
      Ok(HealthResponse(
        status       = "healthy",
        model_loaded = model.isLoaded,
        num_classes  = Config.NUM_CLASSES,
        classes      = Config.CLASS_NAMES.toList
      ).asJson)

    // POST /predict  (multipart/form-data, champ "file")
    case req @ POST -> Root / "predict" =>
      req.decode[multipart.Multipart[IO]] { multipart =>
        val filePart = multipart.parts.find(_.name.contains("file"))
        filePart match {
          case None => BadRequest("Champ 'file' manquant")
          case Some(part) =>
            for {
              imageBytes <- part.body.compile.toVector.map(_.toArray)
              response   <- IO {
                val img          = ImageIO.read(new ByteArrayInputStream(imageBytes))
                val original_size = List(img.getWidth, img.getHeight)
                val pixels       = InferenceHelpers.preprocess_image(imageBytes)

                val detections: List[Detection] = model.run(pixels) match {
                  case None => List.empty
                  case Some((cls_preds, reg_preds)) =>
                    val dets = InferenceHelpers.decode_predictions(cls_preds, reg_preds, model.anchors)
                    // Convertir bbox normalisé en pixels
                    dets.map { d =>
                      d.copy(bbox_pixels = List(
                        (d.bbox(0) * original_size(0)).toInt,
                        (d.bbox(1) * original_size(1)).toInt,
                        (d.bbox(2) * original_size(0)).toInt,
                        (d.bbox(3) * original_size(1)).toInt
                      ))
                    }
                }

                val has_dangerous   = detections.exists(_.is_dangerous)
                val dangerous_items = detections.filter(_.is_dangerous).map(_.`class`)

                PredictResponse(
                  filename        = part.filename.getOrElse("upload"),
                  image_size      = original_size,
                  detections      = detections,
                  num_detections  = detections.size,
                  has_dangerous   = has_dangerous,
                  dangerous_items = dangerous_items,
                  summary         = s"${detections.size} objet(s) détecté(s)" +
                    (if (has_dangerous) s", ALERTE: ${dangerous_items.mkString(", ")}" else "")
                )
              }
              result <- Ok(response.asJson)
            } yield result
        }
      }
  }

  // -------------------------------------------------------------------------
  // Entrée principale
  // -------------------------------------------------------------------------
  override def run: IO[Unit] = {
    val spark = SparkSession.builder()
      .appName("XrayVision-InferenceService")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // Enregistrement de la UDF pour l'inférence batch Spark
    SparkInference.registerUDF(spark, Config.MODEL_PATH)

    println("""
      ╔═══════════════════════════════════════════════════════════════╗
      ║     XRAYVISION — INFERENCE SERVICE (Spark + Scala + ONNX)    ║
      ║     Endpoints: GET /health   POST /predict                    ║
      ╚═══════════════════════════════════════════════════════════════╝
    """)

    BlazeServerBuilder[IO]
      .bindHttp(8000, "0.0.0.0")
      .withHttpApp(routes.orNotFound)
      .serve
      .compile
      .drain
  }
}