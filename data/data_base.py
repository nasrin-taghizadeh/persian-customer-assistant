import os
import shutil
import sqlite3
import pandas as pd
import requests


db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
# The backup lets us restart for each tutorial section
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"


def load_travel_db(data_dir):
    local_db = os.path.join(data_dir, local_file)

    overwrite = False
    if overwrite or not os.path.exists(local_db):
        response = requests.get(db_url)
        response.raise_for_status()  # Ensure the request was successful
        with open(local_db, "wb") as f:
            f.write(response.content)
        # Backup - we will use this to "reset" our DB in each section
        shutil.copy(local_db, backup_file)
    # Convert the flights to present time for our tutorial
    conn = sqlite3.connect(local_db)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
            pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
            + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
                pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    db = local_file  # We'll be using this local file as our DB in this tutorial
    return os.path.join(data_dir, db)
