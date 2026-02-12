"""ML pipeline for Illinois General Assembly legislative intelligence.

Transforms cached legislative JSON data into a normalized Parquet star schema,
then provides interactive human-in-the-loop training for entity resolution,
bill outcome prediction, and anomaly detection.
"""
