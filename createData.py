import duckdb
import random 
from random import randrange
import pandas as pd

from duckdb.typing import *
from faker import Faker
import hashlib



def random_ssn(n):
    return str(n).zfill(10)

def user_hash(n):
    return hashlib.sha256(str.encode(str(n).zfill(10))).hexdigest()



def random_clinical_trial_outcome(n):
    i=random.randrange(0, 3)
    return i


duckdb.create_function("user_hash_from_ssn", user_hash, [BIGINT], VARCHAR)
duckdb.create_function("ssn", random_ssn, [BIGINT], VARCHAR)
duckdb.create_function("clinical_trial_outcome", random_clinical_trial_outcome, [DOUBLE], BIGINT)

res = duckdb.sql("COPY (SELECT ssn(i) as national_id, user_hash_from_ssn(i) as user_hash, clinical_trial_outcome(random()) as clinical_trial_outcome FROM generate_series(1, 100000) s(i)) TO 'data/outcome.parquet'  (FORMAT 'parquet')")
