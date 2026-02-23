object Config {

  val IMAGE_SIZE = 256

  //ratios de split
  val TRAIN_RATIO = 0.7
  val VAL_RATIO = 0.15
  val TEST_RATIO = 0.15
  
  val RANDOM_SEED = 42
  
  //paths
  val RAW_DATA_PATH = "../data/RawData"
  val OUTPUT_PATH = "../data/processed"
  
  //mapping label <-> id
  val labelToId: Map[String, Int] = Map(
    "Gun" -> 0,
    "Knife" -> 1,
    "Scissors" -> 2,
    "Bullet" -> 3,
    "Razor_blade" -> 4,
    "Shuriken" -> 5,
    "Lighter" -> 6,
    "Pressure_vessel" -> 7,
    "Wrench" -> 8,
    "Pliers" -> 9,
    "Hammer" -> 10,
    "Screwdriver" -> 11,
    "Battery" -> 12,
    "Bat" -> 13,
    "Saw_blade" -> 14,
    "Fireworks" -> 15,
    "Dart" -> 16
  )
  
  val idToLabel: Map[Int, String] = labelToId.map(_.swap)
}
