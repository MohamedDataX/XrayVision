// =============================================================================
// XRAYVISION - Preprocessing avec Apache Spark
// Détection d'objets interdits dans les images X-ray (17 classes)
// =============================================================================

lazy val root = (project in file("."))
  .settings(
    name := "xray-preprocessing-spark",
    version := "1.0",
    scalaVersion := "2.12.18",
    
    // Main class
    Compile / mainClass := Some("PreprocessingMain"),
    
    // Dépendances Spark + JSON
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.5.0",
      "org.apache.spark" %% "spark-sql" % "3.5.0",
      "commons-io" % "commons-io" % "2.15.1"
    ),
    
    // Assembly settings
    assembly / assemblyJarName := "xray-preprocessing.jar",
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case _ => MergeStrategy.first
    },
    
    // JVM pour Mac CPU + Java 17 compatibility
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq(
      "-Xmx4g",
      "-Xms1g",
      "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
      "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
    )
  )