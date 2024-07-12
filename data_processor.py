from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import jsonlines

def chunk_by_topic(text, n_topics=5):
    sentences = text.split('.')
    if len(sentences) < n_topics:
        return [text]  # Return the entire text as a single chunk if there are fewer sentences than topics
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(sentences)
    
    lda = LatentDirichletAllocation(n_components=min(n_topics, doc_term_matrix.shape[0]), random_state=42)
    lda.fit(doc_term_matrix)
    
    chunks = [[] for _ in range(lda.n_components)]
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            topic = lda.transform(doc_term_matrix[i:i+1]).argmax()
            chunks[topic].append(sentence)
    
    return [' '.join(chunk) for chunk in chunks if chunk]

def process_and_store_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
    ]
    schema = CollectionSchema(fields, "NVIDIA docs collection")
    collection = Collection("nvidia_docs", schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 500},
    }
    collection.create_index("embedding", index_params)

    with jsonlines.open('nvidia_cuda_docs.jsonl', 'r') as reader:
        scraped_data = list(reader)

    for item in scraped_data:
        chunks = chunk_by_topic(item['content'])
        for chunk in chunks:
            embedding = model.encode(chunk)
            collection.insert([
                [embedding.tolist()],
                [item['url']]
            ])

    collection.flush()
