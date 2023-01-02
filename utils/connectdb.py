from sqlalchemy import create_engine
import json
import psycopg2
import psycopg2.extras

def create_db_connection():
    f = open(r'C:\Users\trent\VSCode\edge\src\env\config.json')
    data = json.load(f)
    f.close()
    password = data['db'][0]['password']
    database = data['db'][0]['database']
    engine = create_engine('postgresql://postgres:' + password + '@localhost:5432/' + database)
    conn = engine.connect().execution_options(stream_results=True)
    return conn

class psycopg_connection():
    def pgconnect(self):
        try:
            self.conn = psycopg2.connect(host='localhost',
                                    database='asxdata',
                                    user='postgres',
                                    password='Blacksmif6')
            print('connected')
        except Exception as e:
            print("unable to connect to the database")
            print(e)
        return self.conn

    def pgquery(self, conn, sqlcmd, args, silent=False, returntype='tuple'):
       """ utility function to execute some SQL query statement
           it can take optional arguments (as a dictionary) to fill in for placeholder in the SQL
           will return the complete query result as return value - or in case of error: None
           error and transaction handling built-in (by using the 'with' clauses) """
       retval = None
       with self.conn:
          cursortype = None if returntype != 'dict' else psycopg2.extras.RealDictCursor
          with self.conn.cursor(cursor_factory=cursortype) as cur:
             try:
                if args is None:
                    cur.execute(sqlcmd)
                else:
                    cur.execute(sqlcmd, args)
                retval = cur.fetchall() # we use fetchall() as we expect only _small_ query results
             except Exception as e:
                if e.pgcode != None and not(silent):
                    print("db read error: ")
                    print(e)
       return retval

    def disconnect(self):
        self.conn.close()
        print('connection closed')


