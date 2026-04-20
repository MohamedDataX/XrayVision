package com.xrayvision.service

import org.http4s._
import org.http4s.dsl.io._
import org.http4s.circe._
import io.circe.syntax._
import io.circe.generic.auto._
import cats.effect.IO
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import com.xrayvision.config.Config
import com.xrayvision.core.{OnnxModelWrapper, InferenceHelpers}
import com.xrayvision.models.{Detection, HealthResponse, PredictResponse}


class Routes(model: OnnxModelWrapper) {

  val routes = HttpRoutes.of[IO] {

    // GET /health
    case GET -> Root / "health" =>
      Ok(HealthResponse(
        status = "healthy",
        model_loaded = model.isLoaded,
        num_classes = Config.NUM_CLASSES,
        classes = Config.CLASS_NAMES.toList
      ).asJson)

    // POST /predict (multipart/form-data, champ "file")
    case req @ POST -> Root / "predict" =>
      req.decode[multipart.Multipart[IO]] { multipart =>
        val filePart = multipart.parts.find(_.name.contains("file"))
        filePart match {
          case None => BadRequest("Champ 'file' manquant")
          case Some(part) =>
            for {
              imageBytes <- part.body.compile.toVector.map(_.toArray)
              response <- IO {
                val img = ImageIO.read(new ByteArrayInputStream(imageBytes))
                val original_size = List(img.getWidth, img.getHeight)
                val pixels = InferenceHelpers.preprocess_image(imageBytes)

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

                val has_dangerous = detections.exists(_.is_dangerous)
                val dangerous_items = detections.filter(_.is_dangerous).map(_.`class`)

                PredictResponse(
                  filename = part.filename.getOrElse("upload"),
                  image_size = original_size,
                  detections = detections,
                  num_detections = detections.size,
                  has_dangerous = has_dangerous,
                  dangerous_items = dangerous_items,
                  summary = s"${detections.size} objet(s) détecté(s)" +
                    (if (has_dangerous) s", ALERTE: ${dangerous_items.mkString(", ")}" else "")
                )
              }
              result <- Ok(response.asJson)
            } yield result
        }
      }
  }
}

object Routes {
  def apply(model: OnnxModelWrapper): Routes = new Routes(model)
}
