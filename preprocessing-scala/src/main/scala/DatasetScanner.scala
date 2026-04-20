import java.io.File
import scala.io.Source

object DatasetScanner {
  
  private val extensions = Set("jpg", "jpeg", "png", "bmp")
  
//Scanner les samples + (avec annotations JSON)
  def scanPositiveSamples(dir: File): Seq[Sample] = {
    val imagesDir = new File(dir, "images")
    val labelsDir = new File(dir, "labels")
    
    if (!imagesDir.exists() || !labelsDir.exists()) return Seq.empty
    
    labelsDir.listFiles()
      .filter(_.getName.endsWith(".json"))
      .flatMap(parseJsonFile(_, imagesDir))
      .toSeq
  }
  
// same pour les samples négatifs
  def scanNegativeSamples(dir: File): Seq[Sample] = {
    val imagesDir = new File(dir, "images")
    if (!imagesDir.exists()) return Seq.empty
    
    imagesDir.listFiles()
      .filter(f => f.isFile && hasValidExtension(f.getName))
      .map(f => Sample(f.getAbsolutePath, List.empty))
      .toSeq
  }
  
  private def hasValidExtension(name: String): Boolean =
    extensions.exists(ext => name.toLowerCase.endsWith(s".$ext"))
  
  private def parseJsonFile(jsonFile: File, imagesDir: File): Option[Sample] = {
    try {
      val baseName = jsonFile.getName.replace(".json", "")
      val imageFile = new File(imagesDir, s"${baseName}_OL.png")
      
      if (!imageFile.exists()) return None
      
      val json = Source.fromFile(jsonFile).mkString
      val objects = parseAnnotations(json)
      
      if (objects.nonEmpty) Some(Sample(imageFile.getAbsolutePath, objects))
      else None
    } catch {
      case _: Exception => None
    }
  }
  
// Parser les annotations JSON
  private def parseAnnotations(json: String): List[ObjectAnnotation] = {
    val objectsPattern = """"objects"\s*:\s*\[(.*?)\]""".r
    val labelPattern = """"label"\s*:\s*"([^"]+)"""".r
    val olBbPattern = """"ol_bb"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]""".r
    
    objectsPattern.findFirstMatchIn(json).map { m =>
      m.group(1).split("\\}")
        .filter(_.contains("label"))
        .flatMap { block =>
          for {
            labelMatch <- labelPattern.findFirstMatchIn(block)
            bbMatch <- olBbPattern.findFirstMatchIn(block)
            classId = Config.labelToId.getOrElse(labelMatch.group(1), -1)
            if classId >= 0
          } yield ObjectAnnotation(
            labelMatch.group(1),
            classId,
            (bbMatch.group(1).toInt, bbMatch.group(2).toInt, 
             bbMatch.group(3).toInt, bbMatch.group(4).toInt)
          )
        }.toList
    }.getOrElse(List.empty)
  }
}
