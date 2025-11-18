# Detexify Training Data

    gunzip -c detexify.sql.gz | psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER detexify 

See `symbols.json` for the symbol database. The „key“ column in the postgres table maps to the „id“ field in the JSON.

The „strokes“ column contains an array of strokes each of which contains an array of [x,y,t] triples. [[[x1,y1,t1], [x2,y2,t2], …], …].

Have fun!
