import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.HashPartitioner


object testing {

  import org.apache.spark.sql.SparkSession

  val spark: SparkSession =
    SparkSession.
      builder().
      appName("user_based_cf").
      config("spark.master", "local").
      getOrCreate()

  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val user_df = read("/Users/xyyang/Downloads/ml-latest-small/ratings.csv")
    val test_df = read("/Users/xyyang/Downloads/testing_small.csv")

    user_df.createOrReplaceTempView("raw")
    test_df.createOrReplaceTempView("test")

    val train_df = spark.sql("with temp as(select userId, movieId from raw except select * from test) " +
      "select raw.userId, raw.movieId,rating from raw Join temp On (temp.userId=raw.userId AND temp.movieId=raw.movieId)")
    train_df.createOrReplaceTempView("train")

    spark.catalog.dropTempView("test")
    spark.catalog.dropTempView("raw")

    val test = read("/Users/xyyang/Desktop/test/ratesAndPred_all/test_1.csv")
    test.createOrReplaceTempView("testing")

    val pearson_rdd = spark.sparkContext.textFile("/Users/xyyang/Documents/Scala/movie-rating/ml-latest-small/pearson/part-00000").
      map(line => line.stripPrefix("((").stripSuffix(")")).
      map(_.split(',')).
      map { case Array(user, movie, p) => ((user.toInt, movie.stripSuffix(")").toInt), p.toDouble) }.
      cache()


    val average = spark.sql("SELECT userId, AVG(rating) as average FROM train Group By userId").
      rdd.map { case Row(userId, avg) => (userId.asInstanceOf[String].toInt, avg.asInstanceOf[Double]) }.
      cache()

    val user_movie = spark.sql("with average as (SELECT userId, AVG(rating) as avg FROM train Group By userId)" +
      "select train.userId,movieId,rating - avg from train Join average On average.userId = train.userId").
      rdd.map { case Row(userId, movieId, r_avg) =>
      ((userId.asInstanceOf[String].toInt, movieId.asInstanceOf[String].toInt), r_avg.asInstanceOf[Double])
    }.
      cache()

    val test_rdd = spark.sql("WITH average as(SELECT userId, AVG(rating) as avg FROM train Group By userId)" +
      "select testing.userId, movieId, avg from testing Join average On average.userId = testing.userId").
      rdd.map { case Row(userId, movieId, avg) =>
      ((userId.asInstanceOf[String].toInt, movieId.asInstanceOf[String].toInt), avg.asInstanceOf[Double])}.
      cache()



    //average.repartition(1).sortBy(_._1).saveAsTextFile("average")

    //neighbors
    val test_user = test_rdd.map { case ((user, movie), avg) => user }.collect.toSet
    /*
    val neighbor_rdd = pearson_rdd.filter{case((u1,u2),pearson) => test_user.intersect(Set(u1)).nonEmpty}.
      filter{case((u1,u2),pearson) => pearson > 0.5}.
      sortBy{case((k1,k2),v) => k2}.
      sortBy{case((k1,k2),v) => k1}

    neighbor_rdd.
      repartition(1).
      saveAsTextFile("test_neighbor_0.5")
*/


    prediction(user_movie, test_rdd, pearson_rdd)


  }

  def read(resource: String): DataFrame = {
    spark.read.option("header", "true").
      csv(resource) //List[ANY,ANY,ANY,ANY]
  }



  // prediction for a bunch of user, movie pair, returns the [(userId,movieId),(actual,predicted)]
  def prediction(user_movie: RDD[((Int, Int), Double)],
                 test_rdd: RDD[((Int, Int), Double)],
                 neighbor_pearson: RDD[((Int, Int), Double)]) = {

    val movie_based_ratingMap = collection.mutable.Map(user_movie.collectAsMap().toSeq: _*) //key:(user,movie) value: rating-avg
    val pearsonMap = collection.mutable.Map(neighbor_pearson.collectAsMap().toSeq: _*)

    val predict = test_rdd.
      partitionBy(new HashPartitioner(200)).
      map{ case ((target_u, target_m), avg_a) =>
          ((target_u, target_m),
            calcu_helper(movie_based_ratingMap.filter { case ((user, movie), v) => movie == target_m }, //r_u_i - avg_u
              pearsonMap.filter { case ((u1, u2), p) => u1 == target_u }, // pearson of target user and all other users
              avg_a))
    }

    predict.foreach(println)
    /*
    validation.
      partitionBy(new HashPartitioner(100)).
      join(predict).
      cache()
      */
  }

  def calcu_helper(r_u_i_avg: collection.mutable.Map[(Int, Int), Double],
                   pearsonMap: collection.mutable.Map[(Int, Int), Double],
                   avg_a: Double): Double = {
    var result = 0.0
    var division_sum = 0.0
    if (pearsonMap.isEmpty) {
      //cannot predict caused by zero-neighbor
      avg_a
    }
    else {
      var other_user = Set.empty[Int]
      var target_movie = 0
      for (((user, movie), r) <- r_u_i_avg) {
        other_user += user
        target_movie = movie
      }

      val pearson = pearsonMap.filter { case ((u1, u2), p) => other_user.intersect(Set(u2)).nonEmpty } // filter on users who rated i

      println("selected neighbor pearson: for item " + target_movie)
      pearson.foreach(println)

      for (((user, movie), rate_avg) <- r_u_i_avg) {
        if (pearson.filter { case ((u1, u2), p) => u2 == user }.values.isEmpty) {
          //users who rated i doesn't appear as neighbor
          avg_a
        }
        else {
          result = result + rate_avg * pearson.filter { case ((u1, u2), p) => u2 == user }.values.toList(0)
          division_sum = division_sum + Math.abs(pearson.filter { case ((u1, u2), p) => u2 == user }.values.toList(0))
        }
      }

      println("denominator: " + result)
      println("division_sum: "+ division_sum)
      println("avg: " + avg_a)

      if (division_sum == 0) avg_a
      else if (result / division_sum + avg_a < 0) {
          0.0
      }
      else if (result / division_sum + avg_a > 5) {
        5.0
      }
      else result / division_sum + avg_a
    }
  }
}