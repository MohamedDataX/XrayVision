import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}


// Traitement d'image : conversion grayscale et resize
object ImageProcessing {
  
  // Convertir en grayscale et resize à la taille configurée
  def process(img: BufferedImage, size: Int): BufferedImage = {
    val gray = new BufferedImage(size, size, BufferedImage.TYPE_BYTE_GRAY)
    val g2d: Graphics2D = gray.createGraphics()
    
    g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
    g2d.drawImage(img, 0, 0, size, size, null)
    g2d.dispose()
    
    gray
  }
  
  // Convertir bbox -> format YOLO normalisé
  def bboxToYolo(obj: ObjectAnnotation, origWidth: Int, origHeight: Int): String = {
    val (xMin, yMin, xMax, yMax) = obj.bbox
    
    val xCenter = clamp(((xMin + xMax) / 2.0) / origWidth)
    val yCenter = clamp(((yMin + yMax) / 2.0) / origHeight)
    val width = clamp((xMax - xMin).toDouble / origWidth)
    val height = clamp((yMax - yMin).toDouble / origHeight)
    
    f"${obj.classId} $xCenter%.6f $yCenter%.6f $width%.6f $height%.6f"
  }
  
  private def clamp(v: Double): Double = math.max(0, math.min(1, v))
}
