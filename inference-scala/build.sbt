name := "xrayvision-inference"
version := "1.0"
scalaVersion := "2.13.12"

mainClass in Compile := Some("com.xrayvision.Main")

val sparkVersion = "3.5.0"
val http4sVersion = "0.23.27"
val circeVersion = "0.14.6"

libraryDependencies ++= Seq(
  // Spark
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql"  % sparkVersion,
  //"org.http4s" %% "http4s-blaze-server" % "0.23.23",
  "org.http4s" %% "http4s-ember-server" % http4sVersion,

  // ONNX Runtime
  "com.microsoft.onnxruntime" % "onnxruntime" % "1.17.0",

  // HTTP4s — EmberServer (nouveau, non-déprécié)
  "org.http4s" %% "http4s-ember-server" % http4sVersion,
  "org.http4s" %% "http4s-circe"        % http4sVersion,
  "org.http4s" %% "http4s-dsl"          % http4sVersion,

  // Circe
  "io.circe" %% "circe-generic" % circeVersion,
  "io.circe" %% "circe-parser"  % circeVersion,

  // Cats Effect
  "org.typelevel" %% "cats-effect" % "3.5.0",

  // IP4s (pour Host/Port dans EmberServerBuilder)
  "com.comcast" %% "ip4s-core" % "3.3.0"
)

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case "reference.conf"         => MergeStrategy.concat
  case _                        => MergeStrategy.first
}

assembly / mainClass := Some("com.xrayvision.Main")

Compile / run / fork := true

javaOptions ++= Seq(
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "-Dcats.effect.warnOnNonMainThreadDetected=false"
)
