from pymilvus import connections, Collection
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_expansion(query):
    tokens = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    expanded_tokens = filtered_tokens.copy()
    for token in filtered_tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                if lemma.name() not in expanded_tokens:
                    expanded_tokens.append(lemma.name())
    
    domain_specific_terms = ['cuda', 'nvidia', 'gpu', 'parallel', 'computing', 'acceleration']
    expanded_tokens.extend(domain_specific_terms)
    
    expanded_query = ' '.join(expanded_tokens)
    return expanded_query

def fetch_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content if content else "No content found on the page."
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def hybrid_retrieval(query, collection, top_k=20):
    expanded_query = query_expansion(query)
    
    results = collection.query(
        expr="id >= 0",
        output_fields=["id", "url", "embedding"],
        limit=collection.num_entities
    )
    
    urls = [item['url'] for item in results]
    embeddings = [item['embedding'] for item in results]
    
    # BM25 retrieval
    bm25 = BM25Okapi(urls)
    bm25_scores = bm25.get_scores(expanded_query.split())
    
    # Semantic retrieval
    query_embedding = sentence_model.encode(expanded_query)
    semantic_scores = util.dot_score(query_embedding, embeddings).squeeze().tolist()
    
    # Combine scores
    combined_scores = [0.3 * bm25 + 0.7 * sem for bm25, sem in zip(bm25_scores, semantic_scores)]
    
    # Get top-k results
    top_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:top_k]
    
    contexts = []
    for i in top_indices:
        content = fetch_content(urls[i])
        if content:
            contexts.append((urls[i], content, combined_scores[i]))

    return contexts

def re_rank(query, contexts, top_k=5):
    query_embedding = sentence_model.encode(query)
    
    re_ranked = []
    for url, content, initial_score in contexts:
        sentences = sent_tokenize(content)
        sentence_embeddings = sentence_model.encode(sentences)
        
        max_similarity = max(util.dot_score(query_embedding, sentence_embeddings).squeeze().tolist())
        
        # Combine initial score with max similarity
        final_score = 0.5 * initial_score + 0.5 * max_similarity
        re_ranked.append((url, content, final_score))
    
    # Sort by final score and return top-k
    re_ranked.sort(key=lambda x: x[2], reverse=True)
    return re_ranked[:top_k]
