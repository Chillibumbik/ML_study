from Libs.DBFiller import make_index_and_fill

from pathlib import Path
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    # http_auth=(),
    use_ssl=False
)

make_index_and_fill(client, Path("data"), index_name="docs")

