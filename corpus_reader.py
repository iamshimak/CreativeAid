import os
from models.models import Corpus


class CorpusReader:
    def __init__(self, root, fields, encoding=None):
        self.root = root
        self.fields = fields
        self.encoding = encoding

    def files(self):
        return self.CorpusIterator(self, self.files_details())

    def files_details(self):
        file_col = []
        for root, dirs, files in os.walk(self.root):
            for name in files:
                file = Corpus(name, os.path.join(root, name))
                file_col.append(file)
        return file_col

    class CorpusIterator:
        def __init__(self, corpus_reader, files):
            self.current = 0
            self.length = len(files)
            self.corpus_reader = corpus_reader
            self.files = files

        def __iter__(self):
            return self

        def __next__(self):
            if self.current >= self.length:
                raise StopIteration
            else:
                file = self.files[self.current]
                file.contents = file.get_contents()
                self.current += 1
                return file

        # def __getitem__(self):


if __name__ == '__main__':
    corpus_reader = CorpusReader("../Test", "")
    files = corpus_reader.files()
    for file in files:
        print(file.contents)
    print(corpus_reader.files_details())
