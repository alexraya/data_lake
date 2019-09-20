import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import DateType
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import date_format
import pyspark.sql.functions as F


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.selectExpr(
    "song_id",
    "title",
    "artist_id",
    "year",
    "duration").orderBy("song_id").drop_duplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs'))

    # extract columns to create artists table
    artists_table = df.selectExpr(
    "artist_id",
    "artist_name as name",
    "artist_location as location",
    "artist_latitude as latitude",
    "artist_longitude as longitude").orderBy("artist_id").drop_duplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists'))


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where("page = 'NextSong'")

    # extract columns for users table    
    users_table = df.selectExpr(
    "userId as user_id",
    "firstName as first_name",
    "lastName as last_name",
    "gender",
    "level").filter("user_id <> ''").orderBy("user_id").drop_duplicates()
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn('timestamp', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
    df = df.withColumn('datetime', get_datetime('ts'))
    
    # extract columns to create time table
    time_table = df.select(
    col('timestamp').alias('start_time'),
    hour('timestamp').alias('hour'),
    dayofmonth('timestamp').alias('day'),
    weekofyear('timestamp').alias('week'),
    month('timestamp').alias('month'),
    year('timestamp').alias('year'),
    date_format('timestamp', 'u').alias('weekday')).orderBy("start_time").drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'time'))

    # read in song data to use for songplays table
    song_df = spark.read.json(os.path.join(input_data, 'song_data/*/*/*/*.json')).selectExpr(
    "song_id",
    "title",
    "artist_id",
    "artist_name",
    "year",
    "duration").drop_duplicates()

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(
    song_df,
    (df.song == song_df.title) & 
    (df.artist == song_df.artist_name) &
    (df.length == song_df.duration) &
    (year(df.timestamp) == song_df.year),
    'left_outer').select(
    df.timestamp.alias("start_time"),
    df.userId.alias("user_id"),
    df.level,
    song_df.song_id,
    song_df.artist_id,
    df.sessionId.alias("session_id"),
    df.location,
    df.userAgent.alias("user_agent"),
    year(df.timestamp).alias('year'),
    month(df.timestamp).alias('month')).orderBy("start_time", "user_id").withColumn("songplay_id", F.monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'songplays'))


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://alex-raya-udacity-datalake/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()