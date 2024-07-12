from scrapy.crawler import CrawlerProcess
from nvidia_crawler import NvidiaCrawler
from data_processor import process_and_store_data

def run_spider():
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 1,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': 'nvidia_cuda_docs.jsonl'
    })
    process.crawl(NvidiaCrawler)
    process.start()

if __name__ == "__main__":
    run_spider()
    process_and_store_data()
