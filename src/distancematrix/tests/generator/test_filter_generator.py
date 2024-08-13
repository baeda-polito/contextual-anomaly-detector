#  Copyright © Roberto Chiosa 2024.
#  Email: roberto.chiosa@polito.it
#  Last edited: 16/7/2024

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from src.distancematrix.generator.filter_generator import FilterGenerator
from src.distancematrix.generator.filter_generator import _invalid_data_to_invalid_subseq
from src.distancematrix.generator.filter_generator import is_not_finite
from src.distancematrix.tests.generator.mock_generator import MockGenerator


class TestFilterGenerator(TestCase):
    def test_data_points_are_filtered_for_different_query_and_series(self):
        mock_gen = MockGenerator(np.arange(12).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite)

        filter_gen.prepare(
            3,
            np.array([1, np.inf, 3, 4, 5, np.inf]),
            np.array([np.inf, 2, 3, 4, np.inf])
        )

        npt.assert_equal(mock_gen.series, [1, 0, 3, 4, 5, 0])
        npt.assert_equal(mock_gen.query, [0, 2, 3, 4, 0])

    def test_data_points_are_filtered_for_self_join(self):
        mock_gen = MockGenerator(np.arange(9).reshape((3, 3)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite)

        data = np.array([np.inf, 2, 3, 4, np.inf])
        filter_gen.prepare(3, data)

        npt.assert_equal(mock_gen.series, [0, 2, 3, 4, 0])
        self.assertIsNone(mock_gen.query)

    def test_calc_column_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(12, dtype=float).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare(
            3,
            np.array([1, np.inf, 3, 4, 5, 6], dtype=float),
            np.array([1, 2, 3, 4, np.inf], dtype=float)
        )

        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [2, 6, np.inf])
        npt.assert_equal(filter_gen.calc_column(3), [3, 7, np.inf])

    def test_calc_diag_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(12, dtype=float).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare(
            3,
            np.array([1, np.inf, 3, 4, 5, 6], dtype=float),
            np.array([1, 2, 3, 4, np.inf], dtype=float)
        )

        # i i 2 3
        # i i 6 7
        # i i i i
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [np.inf, 6, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(2), [2, 7])
        npt.assert_equal(filter_gen.calc_diagonal(3), [3])


class TestStreamingFilterGenerator(TestCase):
    def test_streaming_data_points_are_filtered_for_different_query_and_series(self):
        mock_gen = MockGenerator(np.arange(12).reshape((3, 4)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare_streaming(3, 6, 5)

        npt.assert_equal(mock_gen.bound_gen.appended_series, [])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [])

        filter_gen.append_series(np.array([0, np.inf, 1, 2]))
        filter_gen.append_query(np.array([np.inf]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 0, 1, 2])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0])

        filter_gen.append_series(np.array([3, 4, 5]))
        filter_gen.append_query(np.array([0, 1, 2, 3, 4, 5]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 0, 1, 2, 3, 4, 5])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 0, 1, 2, 3, 4, 5])

        filter_gen.append_series(np.array([6, 7, np.nan]))
        filter_gen.append_query(np.array([6, 7]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 0, 1, 2, 3, 4, 5, 6, 7, 0])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 0, 1, 2, 3, 4, 5, 6, 7])

    def test_streaming_data_points_are_filtered_for_self_join(self):
        mock_gen = MockGenerator(np.arange(9).reshape((3, 3)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare_streaming(3, 6)

        npt.assert_equal(mock_gen.bound_gen.appended_series, [])

        filter_gen.append_series(np.array([0, 1, 2, 3]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [])

        filter_gen.append_series(np.array([np.nan, np.inf, 4, 5, 6]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 0, 4, 5, 6])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [])

        filter_gen.append_series(np.array([7, 8, 9]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 0, 4, 5, 6, 7, 8, 9])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [])

    def test_streaming_calc_column_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(100, dtype=float).reshape((10, 10)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare_streaming(3, 6, 5)

        filter_gen.append_series(np.array([0, 1, 2, 3, np.inf]))
        filter_gen.append_query(np.array([0, 1, 2]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2])
        npt.assert_equal(filter_gen.calc_column(0), [0])
        npt.assert_equal(filter_gen.calc_column(1), [1])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf])

        filter_gen.append_series(np.array([4, 5, 6]))
        filter_gen.append_query(np.array([3]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3])
        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(3), [5, 15])

        filter_gen.append_series(np.array([7]))
        filter_gen.append_query(np.array([np.inf, 4]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6, 7])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 0, 4])
        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [15, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(3), [16, np.inf, np.inf])

        filter_gen.append_series(np.array([8]))
        filter_gen.append_query(np.array([5, 6]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6, 7, 8])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 0, 4, 5, 6])
        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf, 55])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf, np.inf, 56])
        npt.assert_equal(filter_gen.calc_column(3), [np.inf, np.inf, 57])

        filter_gen.append_query(np.array([7]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6, 7, 8])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 0, 4, 5, 6, 7])
        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, 55, 65])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf, 56, 66])
        npt.assert_equal(filter_gen.calc_column(3), [np.inf, 57, 67])

    def test_streaming_self_join_calc_column_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(100, dtype=float).reshape((10, 10)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare_streaming(3, 6)

        filter_gen.append_series(np.array([0, 1, 2, 3, np.inf]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0])
        npt.assert_equal(filter_gen.calc_column(0), [0, 10, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [1, 11, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf, np.inf, np.inf])

        filter_gen.append_series(np.array([4, 5, 6]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6])
        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf, np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(3), [np.inf, np.inf, np.inf, 55])

        filter_gen.append_series(np.array([7]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6, 7])
        npt.assert_equal(filter_gen.calc_column(0), [np.inf, np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(1), [np.inf, np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_column(2), [np.inf, np.inf, 55, 65])
        npt.assert_equal(filter_gen.calc_column(3), [np.inf, np.inf, 56, 66])

        filter_gen.append_series(np.array([8, 9]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 0, 4, 5, 6, 7, 8, 9])
        npt.assert_equal(filter_gen.calc_column(0), [55, 65, 75, 85])
        npt.assert_equal(filter_gen.calc_column(1), [56, 66, 76, 86])
        npt.assert_equal(filter_gen.calc_column(2), [57, 67, 77, 87])
        npt.assert_equal(filter_gen.calc_column(3), [58, 68, 78, 88])

    def test_streaming_calc_diag_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(100, dtype=float).reshape((10, 10)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare_streaming(3, 6, 5)

        filter_gen.append_series(np.array([0, 1, 2]))
        filter_gen.append_query(np.array([np.inf, 1, 2]))

        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf])

        filter_gen.append_query(np.array([3, 4, 5]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 4, 5])
        npt.assert_equal(filter_gen.calc_diagonal(0), [10])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [20])
        npt.assert_equal(filter_gen.calc_diagonal(-2), [30])

        filter_gen.append_series(np.array([3, 4, np.nan]))
        filter_gen.append_query(np.array([np.inf, 6, 7]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 4, 0])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 4, 5, 0, 6, 7])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [np.inf, np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(2), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(3), [np.inf])

        filter_gen.append_series(np.array([5, 6, 7, 8]))
        filter_gen.append_query(np.array([8]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 4, 0, 5, 6, 7, 8])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 4, 5, 0, 6, 7, 8])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, np.inf, 76])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [np.inf, np.inf, 77])
        npt.assert_equal(filter_gen.calc_diagonal(2), [np.inf, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(3), [np.inf])
        # i i i i
        # i i i i
        # i i . .

        filter_gen.append_series(np.array([9]))
        filter_gen.append_query(np.array([9]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9])
        npt.assert_equal(mock_gen.bound_gen.appended_query, [0, 1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, 76, 87])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, 86])
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [np.inf, 77, 88])
        npt.assert_equal(filter_gen.calc_diagonal(2), [np.inf, 78])
        npt.assert_equal(filter_gen.calc_diagonal(3), [np.inf])
        # i i i i
        # i . . .
        # i . . .

    def test_streaming_self_join_calc_diag_with_invalid_data(self):
        mock_gen = MockGenerator(np.arange(100, dtype=float).reshape((10, 10)))
        filter_gen = FilterGenerator(mock_gen,
                                     invalid_data_function=is_not_finite).prepare_streaming(3, 6)

        filter_gen.append_series(np.array([np.inf, 1, 2]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf])

        filter_gen.append_series(np.array([3, 4, 5]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 4, 5])
        npt.assert_equal(filter_gen.calc_diagonal(0), [np.inf, 11, 22, 33])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [np.inf, 21, 32])
        npt.assert_equal(filter_gen.calc_diagonal(-2), [np.inf, 31])
        npt.assert_equal(filter_gen.calc_diagonal(-3), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [np.inf, 12, 23])
        npt.assert_equal(filter_gen.calc_diagonal(2), [np.inf, 13])
        npt.assert_equal(filter_gen.calc_diagonal(3), [np.inf])

        filter_gen.append_series(np.array([6, 7, np.nan]))
        npt.assert_equal(mock_gen.bound_gen.appended_series, [0, 1, 2, 3, 4, 5, 6, 7, 0])
        npt.assert_equal(filter_gen.calc_diagonal(0), [33, 44, 55, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-1), [43, 54, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-2), [53, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(-3), [np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(1), [34, 45, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(2), [35, np.inf])
        npt.assert_equal(filter_gen.calc_diagonal(3), [np.inf])


class TestHelperMethods(TestCase):
    def test_invalid_data_to_invalid_subseq(self):
        data = np.array([0, 0, 0, 0, 0, 0], dtype=np.bool)
        corr = np.array([0, 0, 0, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([1, 0, 0, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 0, 0, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 1, 0, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 1, 0, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 1, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 1, 1, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 0, 1, 0, 0], dtype=np.bool)
        corr = np.array([0, 1, 1, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 0, 0, 1, 0], dtype=np.bool)
        corr = np.array([0, 0, 1, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 0, 0, 0, 1], dtype=np.bool)
        corr = np.array([0, 0, 0, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([1, 0, 1, 0, 0, 0], dtype=np.bool)
        corr = np.array([1, 1, 1, 0], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)

        data = np.array([0, 0, 1, 0, 1, 0], dtype=np.bool)
        corr = np.array([1, 1, 1, 1], dtype=np.bool)
        npt.assert_equal(_invalid_data_to_invalid_subseq(data, 3), corr)
