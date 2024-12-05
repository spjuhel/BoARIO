import tempfile
import pytest
import numpy as np
import json
from boario.utils.misc import (
    _fast_sum,
    _divide_arrays_ignore,
    TempMemmap,
    CustomNumpyEncoder,
    flatten,
    lexico_reindex,
    sizeof_fmt,
    print_summary,
)
from pymrio import IOSystem
import pandas as pd


def test_divide_arrays_ignore():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 0, 3, 0])
    result = _divide_arrays_ignore(a, b)
    expected = np.array([1.0, np.inf, 1.0, np.inf])
    assert np.array_equal(result, expected)


# def test_temp_memmap():
#     shape = (3, 3)
#     TempMemmap(filename=None, dtype=np.float64, shape=shape, save=False) as mmap:
#         mmap[:] = np.ones(shape)
#         assert np.array_equal(mmap, np.ones(shape))
#     # Temp file should be deleted after use


def test_temp_memmap_save():
    shape = (3, 3)
    with tempfile.NamedTemporaryFile() as tmpfile:
        mmap = TempMemmap(
            filename=tmpfile.name, dtype=np.float64, shape=shape, save=True
        )
        mmap[:] = np.ones(shape)
        assert np.array_equal(mmap, np.ones(shape))


def test_custom_numpy_encoder():
    obj = {
        "int": np.int32(42),
        "float": np.float64(3.14),
        "array": np.array([1, 2, 3]),
    }
    json_str = json.dumps(obj, cls=CustomNumpyEncoder)
    assert json_str == '{"int": 42, "float": 3.14, "array": [1, 2, 3]}'


def test_flatten():
    nested = [1, [2, [3, [4]], 5]]
    result = list(flatten(nested))
    expected = [1, 2, 3, 4, 5]
    assert result == expected


def test_lexico_reindex():
    Z = pd.DataFrame([[1, 2], [3, 4]], index=["b", "a"], columns=["b", "a"])
    Y = pd.DataFrame([[5], [6]], index=["b", "a"], columns=["a"])
    x = pd.Series([7, 8], index=["b", "a"])
    A = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=["b", "a"], columns=["b", "a"])
    io = IOSystem(Z=Z, Y=Y, x=x, A=A)

    sorted_io = lexico_reindex(io)
    assert list(sorted_io.Z.index) == ["a", "b"]
    assert list(sorted_io.Z.columns) == ["a", "b"]
    assert list(sorted_io.Y.index) == ["a", "b"]
    assert list(sorted_io.Y.columns) == ["a"]
    assert list(sorted_io.x.index) == ["a", "b"]
    assert list(sorted_io.A.index) == ["a", "b"]
    assert list(sorted_io.A.columns) == ["a", "b"]


def test_sizeof_fmt():
    assert sizeof_fmt(999) == "999.0B"
    assert sizeof_fmt(1024) == "1.0KiB"
    assert sizeof_fmt(1048576) == "1.0MiB"
    assert sizeof_fmt(1099511627776) == "1.0TiB"


def test_print_summary():
    my_list = [1, 1, 2, 3, 3, 3]
    result = print_summary(my_list)
    expected = ["[1 (x 2), 2, 3 (x 3)] (len: 6, sum: 13)"]
    assert result == expected


def test_print_summary_empty():
    my_list = []
    result = print_summary(my_list)
    assert result == ""


def test_fast_sum_1d_array():
    array = np.array([1, 2, 3])
    result = _fast_sum(array, 0)
    expected = np.sum(array, axis=0)
    assert np.array_equal(result, expected)


def test_fast_sum_2d_array_axis_0():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    result = _fast_sum(array, 0)
    expected = np.sum(array, axis=0)
    assert np.array_equal(result, expected)


def test_fast_sum_2d_array_axis_1():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    result = _fast_sum(array, 1)
    expected = np.sum(array, axis=1)
    assert np.array_equal(result, expected)


def test_fast_sum_3d_array_axis_0():
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = _fast_sum(array, 0)
    expected = np.sum(array, axis=0)
    assert np.array_equal(result, expected)


def test_fast_sum_3d_array_axis_1():
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = _fast_sum(array, 1)
    expected = np.sum(array, axis=1)
    assert np.array_equal(result, expected)


def test_fast_sum_3d_array_axis_2():
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = _fast_sum(array, 2)
    expected = np.sum(array, axis=2)
    assert np.array_equal(result, expected)


def test_invalid_axis():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="Axis out of bounds for input array."):
        _fast_sum(array, 3)


def test_too_many_dimensions_axis_0():
    array = np.random.rand(2, 2, 2, 2)  # 4D array
    with pytest.raises(NotImplementedError, match="Too many dimensions."):
        _fast_sum(array, 0)


def test_too_many_dimensions_axis_1():
    array = np.random.rand(2, 2, 2, 2)  # 4D array
    with pytest.raises(NotImplementedError, match="Too many dimensions."):
        _fast_sum(array, 1)


def test_invalid_negative_axis():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="Axis out of bounds for input array."):
        _fast_sum(array, -1)
