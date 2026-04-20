package com.xrayvision
import com.comcast.ip4s.{Host, Port}
import org.apache.spark.sql.SparkSession
import org.http4s.implicits._
//import org.http4s.blaze.server.BlazeServerBuilder
import org.http4s.ember.server.EmberServerBuilder
import com.comcast.ip4s._
import cats.effect.{IO, IOApp}
import com.xrayvision.config.Config
import com.xrayvision.core.OnnxModelWrapper
import com.xrayvision.service.{SparkInference, Routes}


 //Assemble la configuration + le modèle + Spark +  serveur HTTP

object Main extends IOApp.Simple {

  override def run: IO[Unit] = {
    // init Spark
    val spark = SparkSession.builder()
      .appName(Config.SPARK_APP_NAME)
      .master(Config.SPARK_MASTER)
      .config("spark.driver.memory", Config.SPARK_DRIVER_MEMORY)
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // Charger le modèle ONNX une seule fois (singleton sur le driver)
    val model = new OnnxModelWrapper(Config.MODEL_PATH)

    // Enregistrer la UDF pour l'inférence batch Spark
    SparkInference.registerUDF(spark, Config.MODEL_PATH)

   
    println("""
      ╔═══════════════════════════════════════════════════════════════╗
      ║     XRAYVISION — INFERENCE SERVICE (Spark + Scala + ONNX)    ║
      ║     Endpoints: GET /health   POST /predict                    ║
      ╚═══════════════════════════════════════════════════════════════╝
    """)

    // run serveur HTTP
    val httpApp = Routes(model).routes.orNotFound
   
    EmberServerBuilder.default[IO]

      .withHost(Host.fromString(Config.HTTP_HOST).get)
      .withPort(Port.fromInt(Config.HTTP_PORT).get)

      .withHttpApp(httpApp)
      .build
      .useForever
  }
}
