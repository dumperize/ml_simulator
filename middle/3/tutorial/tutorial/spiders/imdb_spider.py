from pathlib import Path
import jsonlines
import json

import scrapy


class ImdbSpider(scrapy.Spider):
    name = "imdb"
    films = {}

    def start_requests(self):
        urls = [
            'https://www.imdb.com/search/name/?gender=male%2Cfemale&ref_=nv_cel_m',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for a in response.css('h3.lister-item-header a'):
            yield response.follow(a, callback=self.parse_personal_page)

    def parse_personal_page(self, response):
        url = response.url
        name = response.css('h1 span::text').get().strip()
        bio = response.css('.ipc-html-content-inner-div').get()
        movies = response.css('a.ipc-metadata-list-summary-item__t::text').getall()
        born_data = response.css('div[data-testid="birth-and-death-birthdate"] span::text').getall()
        born = None if len(born_data) == 0 else born_data[1]

        for a in response.css('a.ipc-metadata-list-summary-item__t'):
            yield response.follow(a, callback=self.parse_film)

        with jsonlines.open('output.jsonl', mode='a') as writer:
            writer.write({
                'url': url,
                'name': name,
                'born': born,
                'bio': bio,
                'movies': movies
            })
    
    def parse_film(self, response):
        url = response.url.split('?')[0]
        if not self.films.get(url):
            title = response.css('h1::text').get().strip()
            cast = response.css('a[data-testid="title-cast-item__actor"]::text').getall()
            

            with jsonlines.open('films.jsonl', mode='a') as writer:
                writer.write({
                    'url': url,
                    'title': title,
                    'cast': cast,
                })
        else:
            self.films[url] = True
