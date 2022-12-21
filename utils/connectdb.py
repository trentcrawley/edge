from sqlalchemy import create_engine
import json

def create_db_connection():
    f = open(r'C:\Users\trent\VSCode\edge\src\env\config.json')
    data = json.load(f)
    f.close()
    password = data['db'][0]['password']
    database = data['db'][0]['database']
    engine = create_engine('postgresql://postgres:' + password + '@localhost:5432/' + database)
    conn = engine.connect().execution_options(stream_results=True)
    return conn


