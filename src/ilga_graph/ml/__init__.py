"""ML pipeline for Illinois General Assembly legislative intelligence.

Transforms cached legislative JSON data into a normalized Parquet star schema,
then runs fully automated ML analysis:

- **Bill Scoring**: Predicts probability of advancement for every bill
- **Coalition Discovery**: Clusters legislators into voting blocs
- **Anomaly Detection**: Flags suspicious witness slip patterns
- **Action Classification**: Structured categorization of all legislative actions

Run everything with: ``make ml-run``
"""
