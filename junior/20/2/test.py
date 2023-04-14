from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import List, Generator, Tuple
import pandas as pd
import numpy as np
import csv


@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        df = pd.read_csv(self.path)
        return np.ceil(df.shape[0] / self.batch_size).astype(int)

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""

        return [
            self.tokenizer.encode(
                x,
                add_special_tokens=True, max_length=self.max_length)
            for x in batch
        ]

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        with open(self.path) as csv_file:
            headers = csv_file.readline().split(',')
            pos = headers.index('sentiment')

            line_count = 0

            review = []
            sentiment = []
            while line_count < (i+1) * self.batch_size:
                line = csv_file.readline()

                if line_count >= i * self.batch_size:
                    print([csv.DictReader(line)])
                    row_string = line[:-1]
                    pos1 = 0
                    for _ in range(pos):
                        pos1 = row_string.find(',', pos1 + 1)
                    pos2 = row_string.find(',', pos1 + 1)

                    sentiment.append(row_string[pos1 + 1:pos2])
                    review.append(row_string[pos2 + 1:])

                line_count = line_count + 1

            str_to_int = {
                "positive": 1,
                "negative": -1,
                "neutral": 0
            }

            return review, [str_to_int[s] for s in sentiment]

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels


if __name__ == "__main__":
    from transformers import DistilBertModel, DistilBertTokenizer

    MODEL_NAME = 'distilbert-base-uncased'

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    bert = DistilBertModel.from_pretrained(MODEL_NAME)
    d = DataLoader("../data.csv",   tokenizer)
    text, label = d.batch_loaded(0)

    print(len(text))
