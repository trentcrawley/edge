from utils import connectdb
import pandas as pd
import psycopg2
import psycopg2.extras

def create_query(stocklist = [],table = 'asxminutedata', datefrom = '', dateto = '',limit=''):
    if len(stocklist) > 1:
        stockListQryString = str(tuple(stocklist))
    elif len(stocklist) == 1:
        stockListQryString = str(tuple(stocklist)).replace(',)', ')')
    else:
        stocklist = ''
    
    query = "SELECT " + "*" + " FROM " + table
    
    if stocklist != "":
        query = query + " WHERE ticker " + 'in ' + stockListQryString
        if datefrom != "":
            query = query + " AND " + "datetime" + " >= " + "'" + datefrom + "'"
        if dateto != "":
            query = query + " AND " + "datetime" + " <= " + "'" + dateto + "'"
    elif datefrom != "":
        query = query + " WHERE " + "datetime" + " >= " + "'" + datefrom + "'"
        if dateto != "":
            query = query + " AND " + "datetime" + " <= " + "'" + dateto + "'"
    elif dateto != "":
        query = query + " WHERE " + "datetime" + " <= " + "'" + dateto + "'" 
    if limit != "":
        query = query + " LIMIT " + str(limit)
    return query
    
def query_db(stocklist = [],datefrom = '', dateto = '',table = 'asxminutedata',limit='' ):
    '''dates format: 'yyyy-mm-dd'  '''
    conn=connectdb.create_db_connection()
    query = create_query(stocklist=stocklist,table=table,datefrom=datefrom,dateto=dateto,limit=limit)
    df = pd.read_sql(query ,conn,parse_dates= {'datetime':{"format":"%Y-%m-%d %H:%M:S"}},index_col = 'datetime')
    df['date'] = df.index.date
    df['time'] = df.index.time
    conn.close()
    return df


def query_db_chunks(query ='select * from asxminutedata limit 10000000',chunksize=5000000,
    conn=connectdb.create_db_connection()):
    '''loops through db and returns a chunk of data at a time''' 
    
    for df in pd.read_sql(query ,conn , parse_dates= {'datetime':{"format":"%Y-%m-%d %H:%M:S"}},index_col = 'datetime', chunksize=chunksize):
        print(df)

    conn.close()


def querySignalDates(filterlist):
    #once we have datetime ticker concat of relevant days we requery the db for just those days
    query = """
    select subquery.open,high,low,close,volume,value,count,datetime,ticker FROM (select *,concat(datetime::date,ticker) AS 
    tempprimary from asxminutedata) AS subquery where subquery.tempprimary = ANY(%s); 
    """
    dbconn = connectdb.psycopg_connection()
    dbconn.conn = dbconn.pgconnect()
    colnames = ['open', 'high', 'low', 'close', 'volume', 'value', 'count', 'datetime', 'ticker']

    x = dbconn.pgquery(dbconn.conn, query, (filterlist,))
    df = pd.DataFrame(x, columns=colnames)
    df['datetime'] = pd.to_datetime(df['datetime'], format = "%Y-%m-%d %H:%M:S")
    df.set_index('datetime',inplace=True)
    df = df.sort_values(by=['ticker', 'datetime'], ascending=[True, True])
    df['date'] = df.index.date
    df['time'] = df.index.time
    df['value'] = pd.to_numeric(df['value'])
    df['volume'] = pd.to_numeric(df['volume'])
    #df = addfeat(df)
    return df

