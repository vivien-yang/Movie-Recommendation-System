import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.HashPartitioner
import scala.util.control.Breaks._

object User_Based_CF{
  import org.apache.spark.sql.SparkSession
  import org.apache.spark.sql.functions._

  val spark: SparkSession =
    SparkSession.
      builder().
      appName("user_based_cf").
      config("spark.master", "local").
      getOrCreate()


  import spark.implicits._

  def main(args:Array[String]): Unit = {
    val user_df_raw = read("/Users/xyyang/Downloads/ml-latest-small/ratings.csv")//read("/Users/xyyang/Downloads/ml-latest-small/ratings.csv")
    val test_df_raw = read("/Users/xyyang/Downloads/testing_small.csv")//read("/Users/xyyang/Downloads/testing_small.csv")

    // dropping rows with missing values
    val user_df = user_df_raw.na.drop()
    val test_df = test_df_raw.na.drop()

    user_df.createOrReplaceTempView("raw")
    test_df.createOrReplaceTempView("test")

    val train_df = spark.sql("with temp as(select userId, movieId from raw except select * from test) " +
      "select raw.userId, raw.movieId,rating from raw Join temp On (temp.userId=raw.userId AND temp.movieId=raw.movieId)")
    train_df.createOrReplaceTempView("train")

    val validation_rdd = spark.sql("SELECT raw.userId, raw.movieId, rating FROM raw " +
      "JOIN test ON test.userId = raw.userId AND test.movieId = raw.movieId").
      rdd.map{case Row(user,movie,rating) => ((user.asInstanceOf[String].toInt,movie.asInstanceOf[String].toInt),rating.asInstanceOf[String].toDouble)}

    spark.catalog.dropTempView("raw")

/*
    val train_rdd = train_df.
      rdd.map{case Row(userId, movieId, rating) =>
      ((userId.asInstanceOf[String].toInt, movieId.asInstanceOf[String].toInt), rating.asInstanceOf[String].toDouble)}
    val test_rdd = test_df.
      rdd.map{case Row(userId, movieId) => (userId.asInstanceOf[String].toInt, movieId.asInstanceOf[String].toInt)}.
      cache()
    val average = spark.sql("SELECT userId, AVG(rating) as average FROM train Group By userId").
      rdd.map{case Row(userId,avg) => (userId.asInstanceOf[String].toInt, avg.asInstanceOf[Double])}.
      cache()
*/
    // (user,movie),rating - avg
    val user_movie = spark.sql("with average as (SELECT userId, AVG(rating) as avg FROM train Group By userId)" +
      "select train.userId,movieId,rating - avg from train Join average On average.userId = train.userId").
      rdd.map{case Row(userId, movieId, r_avg) =>
      ((userId.asInstanceOf[String].toInt,movieId.asInstanceOf[String].toInt),r_avg.asInstanceOf[Double])}.
      cache()

    // (user,movie), avg_target_user
    val test_rdd = spark.sql("WITH average as(SELECT userId, AVG(rating) as avg FROM train Group By userId)" +
      "select test.userId, movieId, avg from test Join average On average.userId = test.userId").
      rdd.map{case Row(userId, movieId, avg) =>
      ((userId.asInstanceOf[String].toInt, movieId.asInstanceOf[String].toInt), avg.asInstanceOf[Double])}.
      cache()

    val N = spark.sql("SELECT Count(DISTINCT userId) from train").rdd.map{case Row(c) => c.asInstanceOf[Long]}.collect()(0).toInt


    spark.catalog.dropTempView("test")
    spark.catalog.dropTempView("train")
/*
    // RDD[(user1,user2),pearson_corr]
    val pearson_rdd = pearson(user_movie, N).
      sortBy{case((k1,k2),p) => k2}.
      sortBy{case((k1,k2),p) => k1}
    pearson_rdd.
      repartition(1).
      saveAsTextFile("pearson_new")
*/
    //read-in pearson file
    val pearson_rdd = spark.sparkContext.textFile("/Users/xyyang/Documents/Scala/movie-rating/ml-latest-small/pearson/part-00000").
      map(line => line.stripPrefix("((").stripSuffix(")")).
      map(_.split(',')).
      map{case Array(user,movie,p)=> ((user.toInt, movie.stripSuffix(")").toInt),p.toDouble)}.
      cache()


    // predict with filtered users
    //val neighbor_pearson = near_neighbor(pearson_rdd,N)
    //val ratesAndPred = prediction(user_movie, test_rdd, validation_rdd, neighbor_pearson)

    // predict with all users
    val ratesAndPred = prediction(user_movie, test_rdd, validation_rdd, pearson_rdd).
      sortBy{case((k1,k2),(v1,v2)) => k2}.
      sortBy{case((k1,k2),(v1,v2)) => k1}

    ratesAndPred.
      repartition(1).
      saveAsTextFile("ratesAndPred_all")

    display(ratesAndPred)
  }


  // read the csv file into dataframe
  def read(resource: String): DataFrame = {
    spark.read.option("header","true").
      csv(resource)//List[ANY,ANY,ANY,ANY]
  }


  // pre-compute co-rated items for each pair of users
  def pearson(user_movie: RDD[((Int, Int), Double)], N: Int) : RDD[((Int,Int),Double)] ={
    val user_movieMap = user_movie.
      partitionBy(new HashPartitioner(200)).
      map{case((user,movie),r_avg) => (user,(movie,r_avg))}.
      groupByKey().
      collectAsMap()

    var res = Array.empty[(Int,Int)]
    for (i<- 1 to N){
      for (j <- 1 to N){
        if (i != j) res = res :+ (i,j)
      }
    }

    spark.sparkContext.parallelize(res).
      map(line => (line._1,line._2)).
      partitionBy(new HashPartitioner(200)).
      map{case (user1,user2) => ((user1,user2),helper(user_movieMap.get(user1).get,user_movieMap.get(user2).get))}
  }


  def helper(u1_info: Iterable[(Int,Double)], u2_info: Iterable[(Int,Double)]) : Double = {
    val u1_movie_diff = u1_info.toMap //(movie -> r-avg) pair
    val u2_movie_diff = u2_info.toMap

    val movies = u1_movie_diff.keys.toSet.intersect(u2_movie_diff.keys.toSet).toArray //co-rated movies

    if (movies.size <=3 ) { //no co-rated items
      0.0
    }
    else {
      var denominator_sum = 0.0
      var division_a_sum = 0.0
      var division_b_sum = 0.0

      for (i <- 0 until movies.size) {
        val a = u1_movie_diff.get(movies(i)).get
        val b = u2_movie_diff.get(movies(i)).get
        denominator_sum = denominator_sum + a * b
        division_a_sum = division_a_sum + a * a
        division_b_sum = division_b_sum + b * b
      }

      if (division_a_sum != 0 && division_b_sum != 0)
        denominator_sum / Math.sqrt(division_a_sum * division_b_sum)
      else 0.0
    }
  }


  //pick neighbors with pearson > 0.5
  def near_neighbor(pearson_rdd: RDD[((Int,Int),Double)], N: Int) : RDD[((Int,Int),Double)] ={
    pearson_rdd.
      filter{case((u1,u2),pearson) => pearson > 0.5}.
      cache()
  }



  // prediction for a bunch of user, movie pair, returns the [(userId,movieId),(actual,predicted)]
  def prediction(user_movie: RDD[((Int, Int), Double)],
                 test_rdd: RDD[((Int, Int),Double)],
                 validation: RDD[((Int,Int),Double)],
                 pearson: RDD[((Int,Int),Double)]) : RDD[((Int, Int),(Double, Double))] = {

    val movie_based_ratingMap = collection.mutable.Map(user_movie.collectAsMap().toSeq: _*)//key:(user,movie) value: rating-avg
    val pearsonMap = collection.mutable.Map(pearson.collectAsMap().toSeq: _*)

    val predict = test_rdd.
      partitionBy(new HashPartitioner(200)).
      map{case ((target_u,target_m), avg_a) => ((target_u,target_m),
        calcu_helper(movie_based_ratingMap.filter{case ((user,movie),v) => movie == target_m},//r_u_i - avg_u
                      pearsonMap.filter{case((u1,u2),p) => u1 == target_u},// pearson of target user and all other users
                      avg_a))}

    validation.
      partitionBy(new HashPartitioner(100)).
      join(predict).
      cache()
  }


  // prediction calculation
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


      for (((user, movie), rate_avg) <- r_u_i_avg) {
        breakable{
          if (pearson.filter{case ((u1, u2), p) => u2 == user}.values.isEmpty) {//users who rated i doesn't appear as neighbor
            avg_a
          }
          else if (pearson.filter{case ((u1, u2), p) => u2 == user}.values.toList(0) < 0.1) {// filter pearson < 0.1
            break
          }
          else {
            result = result + rate_avg * pearson.filter { case ((u1, u2), p) => u2 == user }.values.toList(0)
            division_sum = division_sum + Math.abs(pearson.filter { case ((u1, u2), p) => u2 == user }.values.toList(0))
          }
        }
      }

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



  def display(ratesAndPred: RDD[((Int,Int),(Double, Double))]): Unit = {
    val count = ratesAndPred.map{case((user,movie),(actual_rate,predicted_rate)) =>
      val count_rate = Math.floor(predicted_rate).toInt
      count_rate match {
        case 0 => 0
        case 1 => 1
        case 2 => 2
        case 3 => 3
        case 4 => 4
        case 5 => 4
        case _ => 0
      }}.
      map(each => (each,1)).
      reduceByKey(_+_).
      rightOuterJoin(spark.sparkContext.parallelize(List((0,0),(1,0),(2,0),(3,0),(4,0)))).
      mapValues{case (v1,v2) => v1.getOrElse(0) + v2}.
      sortBy(_._1)

    val MSE = ratesAndPred.map{case ((user, movie), (r1, r2)) =>
      val err = r1 - r2
      err * err}.
      mean()

    val count_0 = count.filter{case(num,c) => num == 0}.collect()(0)._2
    val count_1 = count.filter{case(num,c) => num == 1}.collect()(0)._2
    val count_2 = count.filter{case(num,c) => num == 2}.collect()(0)._2
    val count_3 = count.filter{case(num,c) => num == 3}.collect()(0)._2
    val count_4 = count.filter{case(num,c) => num == 4}.collect()(0)._2

    println(">=0 and <1: " + count_0)
    println(">=1 and <2: " + count_1)
    println(">=2 and <3: " + count_2)
    println(">=3 and <4: " + count_3)
    println(">=4: " + count_4)
    println("RMSE = " + Math.sqrt(MSE))
  }
}