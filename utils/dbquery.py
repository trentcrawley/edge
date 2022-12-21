from utils import connectdb
import pandas as pd

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


#df = query_db(stocklist=['A2M','BHP'],datefrom='2020-01-01',dateto='2022-01-02')

