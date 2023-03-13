import utils


def test_word_count():
    batch = ['word']
    count = utils.word_count(batch, {})
    assert count.get('word', 0) == 1

    prev_count = {'word': 1}
    count = utils.word_count(batch, prev_count)
    assert count.get('word', 0) == 2


def test_word_count_tricky():
    batch = ['1']
    count = utils.word_count(batch, {})
    assert count['1'] == 1

    prev_count = {1: 1}
    count = utils.word_count(batch, prev_count)
    assert count['1'] == 1
