import utils


def test_word_count():
    batch = ['word']
    prev_count = {}
    count = utils.word_count(batch, prev_count)
    assert count.get('word', 0) == 1


def test_word_count_tricky():
    batch = ['word']
    prev_count = {}
    count = utils.word_count(batch, prev_count)
    assert count.get('word', 0) == 1

    count = utils.word_count(batch, prev_count)
    assert count.get('word', 0) == 1
