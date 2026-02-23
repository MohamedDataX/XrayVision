import java.io.File
import org.apache.commons.io.FileUtils

/**
 * Génération du fichier data.yaml pour l'entraînement YOLO
 */
object YamlGenerator {
  
  def create(): Unit = {
    val yaml = s"""# XrayVision Dataset - Object Detection (17 classes)
# Généré automatiquement par preprocessing-scala

path: ${Config.OUTPUT_PATH}
train: images/train
val: images/val
test: images/test

# Nombre de classes
nc: 17

# Noms des classes
names:
  0: Gun
  1: Knife
  2: Scissors
  3: Bullet
  4: Razor_blade
  5: Shuriken
  6: Lighter
  7: Pressure_vessel
  8: Wrench
  9: Pliers
  10: Hammer
  11: Screwdriver
  12: Battery
  13: Bat
  14: Saw_blade
  15: Fireworks
  16: Dart
"""
    FileUtils.writeStringToFile(new File(s"${Config.OUTPUT_PATH}/data.yaml"), yaml, "UTF-8")
    println("   ✓ data.yaml créé")
  }
}
