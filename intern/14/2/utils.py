def word_count(batch, count={}):
    ''' immutable function word_count '''
    new_count = count.copy()
    for text in batch:
        for word in text.split():
            new_count[word] = new_count.get(word, 0) + 1
    return new_count
