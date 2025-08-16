from Libs import DBIInteractions

from pathlib import Path
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    # http_auth=(),
    use_ssl=False
)


interacter = DBIInteractions.DBInteracter(client=client)



