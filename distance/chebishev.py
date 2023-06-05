from typing import Iterable

from distance.base import BaseSampleDistance


class ChebishevDistance(BaseSampleDistance):
    """Класс для определения расстояния между семплами (Чебышев)"""

    k = 1

    @staticmethod
    def reduction(values: Iterable[float]) -> float:
        return max(values)
