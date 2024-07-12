import scrapy
import time
import json
import re

class NvidiaCrawler(scrapy.Spider):
    name = 'nvidia_crawler'
    start_urls = ['https://docs.nvidia.com/cuda/']
    max_depth = 5
    page_count = 0
    max_pages = 50

    def parse(self, response, depth=1):
        time.sleep(1)  # Sleep for 1 second before each request

        self.page_count += 1

        if self.page_count > self.max_pages:
            self.crawler.engine.close_spider(self, 'Reached maximum number of pages')
            return

        if response.headers.get('Content-Type', b'').startswith(b'application/json'):
            try:
                json_data = json.loads(response.text)
                content = json.dumps(json_data)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON from {response.url}")
                return
        else:
            content = ' '.join(response.css('p::text').getall())

        # Clean the content
        content = self.clean_text(content)

        yield {
            'url': response.url,
            'content': content,
            'depth': depth
        }

        if depth < self.max_depth and self.page_count < self.max_pages:
            if not response.headers.get('Content-Type', b'').startswith(b'application/json'):
                for href in response.css('a::attr(href)').getall():
                    yield response.follow(href, callback=self.parse, cb_kwargs={'depth': depth + 1})

    def clean_text(self, text):
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
