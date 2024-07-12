
# NVIDIA CUDA Documentation QA System

This project implements a complete pipeline for crawling NVIDIA CUDA documentation, processing and storing the data, and providing a question-answering system using hybrid retrieval, re-ranking, and a large language model for answer generation.

## Project Structure

The project consists of the following main components:

1. Web Crawler:
   - `nvidia_crawler.py`: Scrapy spider for crawling NVIDIA CUDA documentation
   - `data_processor.py`: Processes and stores crawled data
   - `main.py`: Orchestrates the crawling and data processing

2. QA System:
   - `data_retrieval.py`: Handles query expansion, content fetching, hybrid retrieval, and re-ranking
   - `llm_interface.py`: Manages the interaction with the LLaMA model for generating answers
   - `streamlit_app.py`: The Streamlit app that provides the user interface and orchestrates the QA process

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/kirito-0512/Web_Crawler/tree/main
   cd nvidia-cuda-qa
   ```

2. Install the required dependencies:
   ```
   pip install scrapy sentence-transformers scikit-learn pymilvus jsonlines streamlit rank_bm25 llama-cpp-python requests beautifulsoup4 nltk
   ```

3. Download the LLaMA model:
   - Obtain the `llama-7b.Q4_K_M.gguf` file (or your preferred GGUF model)
   - Place it in the project root directory

4. Set up Milvus:
   - Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
   - Ensure Milvus is running on localhost:19530

## Usage

### Step 1: Web Crawling and Data Processing

Run the web crawler and data processor:

```
python main.py
```

This will:
1. Crawl the NVIDIA CUDA documentation
2. Process the crawled data
3. Store the processed data in Milvus

### Step 2: Running the QA System

After the crawling and data processing is complete, run the Streamlit app:

```
streamlit run streamlit_app.py
```

Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

## File Descriptions

### Web Crawler Components

#### 1)nvidia_crawler.py

This file contains the Scrapy spider for crawling NVIDIA CUDA documentation:
- Defines the `NvidiaCrawler` class
- Implements the crawling logic, including depth control and content extraction
- Cleans the extracted text

Key methods:
- `parse(self, response, depth=1)`: Parses each page and extracts content
- `clean_text(self, text)`: Cleans the extracted text

#### 2)data_processor.py

This file handles the processing and storage of crawled data:
- Implements topic-based text chunking
- Generates embeddings for the chunks
- Stores the processed data in Milvus

Key functions:
- `chunk_by_topic(text, n_topics=5)`: Chunks text based on topics
- `process_and_store_data()`: Processes crawled data and stores it in Milvus

#### 3)main.py (Crawler)

This file orchestrates the crawling and data processing:
- Sets up and runs the Scrapy crawler
- Calls the data processing function

Key functions:
- `run_spider()`: Configures and runs the Scrapy crawler
- `if __name__ == "__main__":` block: Executes the crawling and processing pipeline

### QA System Components

#### 4)data_retrieval.py

This file contains functions for:
- Query expansion using WordNet and domain-specific terms
- Fetching content from URLs
- Hybrid retrieval combining BM25 and semantic search
- Re-ranking of retrieved documents

Key functions:
- `query_expansion(query)`: Expands the input query
- `fetch_content(url)`: Retrieves the content from a given URL
- `hybrid_retrieval(query, collection, top_k=20)`: Performs hybrid retrieval
- `re_rank(query, contexts, top_k=5)`: Re-ranks the retrieved contexts

#### 5)llm_interface.py

This file handles the interaction with the LLaMA model:
- Loads the GGUF model
- Generates answers based on the query and retrieved contexts

Key function:
- `generate_answer(query, contexts)`: Generates an answer using the LLM

#### 6)streamlit_app.py

The Streamlit app that:
- Connects to Milvus
- Provides a user interface for entering questions
- Orchestrates the QA process (retrieval, re-ranking, answer generation)
- Displays the generated answer and top retrieved documents

## How It All Works Together

1. The web crawler (nvidia_crawler.py) crawls the NVIDIA CUDA documentation.
2. The crawled data is processed and stored in Milvus using data_processor.py.
3. The Streamlit app (streamlit_app.py) provides a user interface for asking questions.
4. When a question is asked:
   - data_retrieval.py performs hybrid retrieval and re-ranking to find relevant documents.
   - llm_interface.py generates an answer using the retrieved contexts and the LLaMA model.
   - The answer and relevant documents are displayed to the user.

## Dependencies

- scrapy
- sentence-transformers
- scikit-learn
- pymilvus
- jsonlines
- streamlit
- rank_bm25
- llama-cpp-python
- requests
- beautifulsoup4
- nltk

## Notes

- Ensure that Milvus is properly set up and running before starting the QA system.
- The LLaMA model file (`llama-7b.Q4_K_M.gguf`) should be placed in the project root directory.
- Adjust the `model_path` in `llm_interface.py` if using a different model file.
- The crawling process may take some time depending on the size of the NVIDIA CUDA documentation.


# Comprehensive Overview of the components.

## 1. **Web Crawler**

The web crawler is implemented using Scrapy. Key components include:

- **NvidiaCrawler Class (subclass of `scrapy.Spider`)**:
  - Defines the starting URL: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
  - Implements depth-limited crawling (max_depth = 5)
  - Limits the total number of pages crawled (optional) for limited search due to hardware limitations and time concerns.
  - Handles both HTML and JSON responses.
  - Extracts text content from paragraphs.
  - Follows links to crawl deeper into the site.

- **CrawlerProcess**:
  - Configures crawler settings (user agent, robots.txt compliance, request rate, output format).
  - Runs the spider and saves output to a JSONLines file.

## 2. **Data Chunking**

The data chunking process uses topic modeling to split large documents into smaller, semantically coherent chunks. Components include:

- **CountVectorizer (from scikit-learn)**:
  - Converts text into a bag-of-words representation.
  - Removes common English stop words.

- **LatentDirichletAllocation (LDA) (from scikit-learn)**:
  - Performs topic modeling on the document.
  - Identifies underlying themes in the text.

- **Custom Chunking Logic**:
  - Splits text into sentences.
  - Assigns each sentence to a topic based on LDA results.
  - Groups sentences by topic to form chunks.

## 3. **Vector Database Creation**

The system uses Milvus, a vector database, to store and efficiently query document embeddings. Components include:

- **SentenceTransformer**:
  - Generates embeddings for text chunks using the 'all-MiniLM-L6-v2' model.

- **Milvus Components**:
  - **Collection**: Defines the structure for storing vectors and metadata.
  - **FieldSchema**: Specifies the data types for each field (id, embedding, url).
  - **HNSW Index**: Enables fast approximate nearest neighbor search.

## 4. **Retrieval**

The retrieval system implements a hybrid approach combining traditional keyword-based search (BM25) and semantic search. Components include:

- **Query Expansion**:
  - Uses WordNet for synonym expansion.
  - Adds domain-specific terms to the query.

- **BM25Okapi (from rank_bm25)**:
  - Implements the BM25 ranking function for keyword-based retrieval.

- **SentenceTransformer**:
  - Generates embeddings for the query and documents.

- **Hybrid Scoring**:
  - Combines BM25 and semantic similarity scores (with weights 0.3 and 0.7 respectively).

## 5. **Re-ranking**

The re-ranking process refines the initial retrieval results for improved relevance. Components include:

- **Sentence-level Similarity**:
  - Splits retrieved documents into sentences.
  - Computes similarity between the query and each sentence.

- **Score Combination**:
  - Combines the initial retrieval score with the maximum sentence-level similarity.
  - Uses equal weights (0.5 each) for initial and sentence-level scores.

## 6. **Question Answering**

The QA component uses a large language model to generate answers based on the retrieved and re-ranked contexts. Components include:

- **GGUF Model (loaded using llama_cpp)**:
  - A quantized language model (e.g., LLaMA 7B) for efficient inference.
  - Configured with a context window of 2048 tokens.

- **Prompt Engineering**:
  - Constructs a prompt combining the query and retrieved contexts.
  - Instructs the model to answer based only on the given context.

- **Answer Generation**:
  - Uses the language model to generate a response.
  - Limits the output to 200 tokens for concise answers.

## System Workflow

These components work together to create a comprehensive QA system:

1. The **crawler** collects NVIDIA CUDA documentation.
2. The **chunking** process breaks large documents into manageable pieces.
3. These chunks are embedded and stored in the **Milvus vector database**.
4. When a query is received, the **retrieval** system combines keyword and semantic search to find relevant documents.
5. The **re-ranking** process refines these results for better relevance.
6. Finally, the **QA** component uses the language model to generate an answer based on the most relevant contexts.


