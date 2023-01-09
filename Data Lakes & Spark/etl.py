import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType, TimestampType
from pyspark.sql.functions import udf, col, year, month, dayofweek, hour, weekofyear, dayofmonth, monotonically_increasing_id
from pyspark.sql import functions as F
from pyspark.sql import types as T
import pandas as pd


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['KEY']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['SECRET']

input_data = "s3a://udacity-dend/"
output_data = "s3a://udacitybucket23456/"

song_data = input_data + "song_data/A/A/A/*.json" # get filepath to song data file
log_data = input_data + "log_data/*/*/*.json"     # get filepath to log data file

def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    
    '''
    Reads JSON song files from input S3, build the songs and artists tables
    and writes the output in parquet format in the destination S3
    
    '''
    song_schema = StructType([
        StructField('artist_id', StringType()),
        StructField('artist_latitude', DoubleType()),
        StructField('artist_location', StringType()),
        StructField('artist_longitude', StringType()),
        StructField('artist_name', StringType()),
        StructField('duration', DoubleType()),
        StructField('num_songs', IntegerType()),
        StructField('title', StringType()),
        StructField('year', IntegerType()),
    ])

    # read song data file & forcing the schema above
    df_song = spark.read.json(song_data, schema=song_schema)
    
    # extract columns to create songs table
    song_fields = ['title', 'artist_id', 'year', 'duration']
    songs_table = df_song.select(song_fields).dropDuplicates().withColumn('song_id', monotonically_increasing_id())

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy('year', 'artist_id').parquet(output_data + 'songs')
    
    # extract columns to create artists table
    artists_fields = ['artist_id', 'artist_name as name', 'artist_location', 'artist_latitude as latitude',
                      'artist_longitude as longitude']
    artists_table = df_song.selectExpr(artists_fields).dropDuplicates()

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data + 'artists')


def process_log_data(spark, input_data, output_data):
    
    '''
    Reads JSON log files from input S3, build the users & time tables
    and writes the output in parquet format in the destination S3
    
    '''
    
    # read log data file
    df_log = spark.read.json(log_data)
    df_log = df_log.filter(df_log.page == 'NextSong') #this is where the relevant log data lies
    
    # extract columns for users table
    users_fields = ['userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level']
    users_table = df_log.selectExpr(users_fields).dropDuplicates()

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + 'users')
    
    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    df_log = df_log.withColumn('timestamp', get_timestamp(df_log.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    df_log = df_log.withColumn('start_time', get_datetime(df_log.timestamp))

    # extract columns to create time table
    df_log = df_log.withColumn('hour', hour('start_time')) \
        .withColumn('day', dayofmonth('start_time')) \
        .withColumn('week', weekofyear('start_time')) \
        .withColumn('month', month('start_time')) \
        .withColumn('year', year('start_time')) \
        .withColumn('weekday', dayofweek('start_time'))

    time_table = df_log.select('start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday','ts')

    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy('year', 'month').parquet(output_data + 'time')
    
    # read in song data to use for songplays table
    df_song = spark.read.parquet(os.path.join(output_data, "songs/*/*/*"))
    songs_logs = df_log.join(df_song, (df_log.song == df_song.title))
    
    # extract columns from joined song and log datasets to create songplays table by joining on time
    artists_df = spark.read.parquet(os.path.join(output_data, "artists"))
    artists_songs_logs = songs_logs.join(artists_df, (songs_logs.artist == artists_df.name))
    
    #dropping ambiguous columns for the next join 
    artists_songs_logs = artists_songs_logs.drop('start_time','month')

    #artists_songs_logs = artists_songs_logs.withColumnRenamed("location", "artists_songs_location")
    songplays = artists_songs_logs.join(time_table, artists_songs_logs.ts == time_table.ts, 'left').drop(artists_songs_logs.year)
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table = songplays.select(
            col('start_time'),
            col('userId').alias('user_id'),
            col('level'),
            col('song_id'),
            col('artist_id'),
            col('sessionId').alias('session_id'),
            col('location'),
            col('userAgent').alias('user_agent'),
            col('year'),
            col('month'),
        ).repartition("year", "month")

    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data + 'songplays')


def main():
    spark = create_spark_session()
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
