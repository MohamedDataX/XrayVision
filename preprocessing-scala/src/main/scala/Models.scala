// Annot objet détecté 

case class ObjectAnnotation(
  label: String,
  classId: Int,
  bbox: (Int, Int, Int, Int) // x_min, y_min, x_max, y_max
) extends Serializable

// sample :  path + annotations
case class Sample(
  imagePath: String,
  objects: List[ObjectAnnotation]
) extends Serializable
