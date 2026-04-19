name := "xrayvision-inference"
version := "1.0"
scalaVersion := "2.13.12"

val sparkVersion = "3.5.0"
val http4sVersion = "0.23.23"
val circeVersion = "0.14.6"

libraryDependencies ++= Seq(
  // Spark


  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql"  % sparkVersion,

  // ONNX Runtime
  "com.microsoft.onnxruntime" % "onnxruntime" % "1.17.0",

  // HTTP4s — blaze est dans un repo séparé depuis 0.23
  "org.http4s" %% "http4s-blaze-server" % "0.23.16",
  "org.http4s" %% "http4s-circe"        % http4sVersion,
  "org.http4s" %% "http4s-dsl"          % http4sVersion,

  // Circe
  "io.circe" %% "circe-generic" % circeVersion,
  "io.circe" %% "circe-parser"  % circeVersion,
)

// Blaze est publié sur un repo distinct
resolvers += "http4s-blaze" at "https://oss.sonatype.org/content/repositories/releases"

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case "reference.conf"         => MergeStrategy.concat
  case _                        => MergeStrategy.first
}


Compile / run / fork := true

javaOptions ++= Seq(
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "-Dcats.effect.warnOnNonMainThreadDetected=false"
)