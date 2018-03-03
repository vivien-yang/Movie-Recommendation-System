/**
  * Created by xyyang on 7/6/17.
  */
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.HashPartitioner

object model_Based_CF{
  def main(args:Array[String]) {
    val sparkConf = new SparkConf().setAppName("Spark_VivienYang")
    val sc = new SparkContext(sparkConf)

    val train_data = sc.textFile(args(0))//training data
    val test_data = sc.textFile(args(1))//test data

    val ratings = train_data.map(_.split(',')).
                      filter{case Array(user, movie, rate,timestamp)=> user!="userId"}.
                      map{case Array(user, movie, rate,timestamp) =>
                            Rating(user.toInt, movie.toInt, rate.toDouble)}.
                      persist()

    // Build the recommendation model using ALS
    val rank = 10
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.01)

    // Evaluate the model on test data
    val test = test_data.map(_.split(',')).
                        filter{case Array(user, movie)=> user!="userId"}.
                        map{case Array(user, movie) => (user.toInt, movie.toInt)}.
                        partitionBy(new HashPartitioner(100)).
                        persist()

    val predictions = model.predict(test).
                            map{case Rating(user, movie, rate) => ((user, movie), rate)}.
                            partitionBy(new HashPartitioner(100)).
                            persist()

    val count = predictions.map{case((user,movie),rate) =>
                                val count_rate = Math.floor(rate).toInt
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
                            sortBy(_._1)
    //count.collect.foreach(println)

    val ratesAndPred = ratings.map{case Rating(user, movie, rate) => ((user, movie), rate)}.
                              partitionBy(new HashPartitioner(100)).
                              join(predictions).
                              persist()

    val MSE = ratesAndPred.map{case ((user, movie), (r1, r2)) =>
                                  val err = r1 - r2
                                  err * err}.
                            mean()

    predictions.map{case((user, movie), rate) => (user,movie,rate)}.
                sortBy(_._2).
                sortBy(_._1).
                repartition(1).
                saveAsTextFile("movie_rating_ml-latest-small")

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
