''' file with test '''
import metrics


def test_profit() -> None:
    ''' test metrics.profit '''
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3


def test_margin() -> None:
    ''' test metrics.margin '''
    assert metrics.margin([1, 2, 3], [1, 1, 1]) == 0.5


def test_markup() -> None:
    ''' test metrics.markup '''
    assert metrics.markup([1, 2, 3], [1, 1, 1]) == 1
