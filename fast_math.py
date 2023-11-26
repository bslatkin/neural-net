import operator
import struct


class BaseTensor:
    def set(self, x, y, value):
        raise NotImplementedError

    def get(self, x, y):
        raise NotImplementedError

    def row(self, y):
        for x in range(self.columns):
            yield self.get(x, y)

    def column(self, x):
        for y in range(self.rows):
            yield self.get(x, y)

    def to_list(self):
        result = []
        for y in range(self.rows):
            result.append(list(self.row(y)))
        return result

    def transpose(self):
        return Transposed(self)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.to_list()!r})'


class Tensor(BaseTensor):
    def __init__(self, columns, rows, *, data=None, type_format='f'):
        self.columns = columns
        self.rows = rows

        if data is None:
            size_bytes = struct.calcsize(type_format)
            data_bytes = bytearray(size_bytes * rows * columns)
        else:
            data_bytes = data.cast('B')
            type_format = data.format

        self.data_bytes = data_bytes
        self.type_format = type_format
        self._init_view()

    def _init_view(self):
        self.view = memoryview(self.data_bytes).cast(
            self.type_format,
            shape=(self.rows, self.columns))

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['view']
        return result

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_view()

    def set(self, x, y, value):
        if y >= self.rows:
            breakpoint()
        if x >= self.columns:
            breakpoint()
        self.view[y, x] = value

    def get(self, x, y):
        if y >= self.rows:
            breakpoint()
        if x >= self.columns:
            breakpoint()
        return self.view[y, x]

    @classmethod
    def from_list(cls, the_list):
        rows = len(the_list)
        columns = len(the_list[0])
        result = cls(columns, rows)
        for y in range(rows):
            for x in range(columns):
                result.set(x, y, the_list[y][x])
        return result


class Transposed(BaseTensor):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    @property
    def columns(self):
        return self.wrapped.rows

    @property
    def rows(self):
        return self.wrapped.columns

    def set(self, x, y, value):
        self.wrapped.set(y, x, value)

    def get(self, x, y):
        return self.wrapped.get(y, x)

    def row(self, y):
        return self.wrapped.column(y)

    def column(self, x):
        return self.wrapped.row(x)

    def transpose(self):
        return self.wrapped


def matrix_multiply(a, b):
    assert a.columns == b.rows, f'{a.columns=} == {b.rows=}'

    result = Tensor(b.columns, a.rows)

    for y in range(result.rows):
        for x in range(result.columns):
            row = a.row(y)
            column = b.column(x)
            total = sum(r * c for r, c in zip(row, column))
            result.set(x, y, total)

    return result


def dot_product(a, b):
    assert a.rows == b.rows == 1, f'{a.rows=} == {b.rows=} == 1'

    result = matrix_multiply(a, b.transpose())

    return result.get(0, 0)


def matrix_elementwise_apply(func, a, b):
    assert a.rows == b.rows, f'{a.rows=} == {b.rows=}'
    assert a.columns == b.columns, f'{a.columns=} == {b.columns=}'

    result = Tensor(a.columns, a.rows)

    for y in range(result.rows):
        for x in range(result.columns):
            a_value = a.get(x, y)
            b_value = b.get(x, y)
            added = func(a_value, b_value)
            result.set(x, y, added)

    return result


def matrix_elementwise_add(a, b):
    return matrix_elementwise_apply(operator.add, a, b)


def matrix_elementwise_multiply(a, b):
    return matrix_elementwise_apply(operator.mul, a, b)


def matrix_elementwise_subtract(a, b):
    return matrix_elementwise_apply(operator.sub, a, b)


def matrix_apply(func, a):
    result = Tensor(a.columns, a.rows)

    for y in range(a.rows):
        for x in range(a.columns):
            input_value = a.get(x, y)
            output_value = func(input_value)
            result.set(x, y, output_value)

    return result


def matrix_rowwise_apply(func, a):
    # Outputs a column vector
    result = Tensor(1, a.rows)

    for y in range(result.rows):
        row = a.row(y)
        value = func(row)
        result.set(0, y, value)

    return result



# def matmul_1x1(a, b):
#     # a = [[x]]
#     # b = [[y]]
#     # result = [[x * y]]
#     return [[a[0][0] * b[0][0]]]

# def matmul_2x2(a, b):
#     columns = len(a)
#     # a = [[2, 3]]
#     # b = [[5]]
#     # result = [[10]]
#     return [[a[0][0] * b[0][0]]]


# def matmul_1x3_3x1(a, b):
#     a_0 = a[0]
#     a_0_0 = a_0[0]
#     a_0_1 = a_0[1]
#     a_0_2 = a_0[2]

#     b_0 = b[0]
#     b_1 = b[1]
#     b_2 = b[2]
#     b_0_0 = b_0[0]
#     b_1_0 = b_1[0]
#     b_2_0 = b_2[0]

#     return [
#         [
#             a_0_0 * b_0_0,
#             a_0_0 * b_1_0,
#             a_0_0 * b_2_0,
#         ],
#         [
#             a_0_1 * b_0_0,
#             a_0_1 * b_1_0,
#             a_0_1 * b_2_0,
#         ],
#         [
#             a_0_2 * b_0_0,
#             a_0_2 * b_1_0,
#             a_0_2 * b_2_0,
#         ],
#     ]



def test_tensors():
    t1 = Tensor(3, 2)
    t1.set(0, 0, 5)
    t1.set(0, 1, 6)
    t1.set(1, 0, 7)
    t1.set(1, 1, 8)
    t1.set(2, 0, 9)
    t1.set(2, 1, 10)
    print(f'{t1=}')

    t2 = t1.transpose()
    print(f'{t2=}')

    t3 = t2.transpose()
    print(f'{t3=}')

    v1 = Tensor(3, 1)
    v1.set(0, 0, 5)
    v1.set(1, 0, 6)
    v1.set(2, 0, 7)
    print(f'{v1=}')

    v2 = Tensor(3, 1)
    v2.set(0, 0, 12)
    v2.set(1, 0, 13)
    v2.set(2, 0, 14)
    print(f'{v2=}')

    v1_dot_v2 = dot_product(v1, v2)
    print(f'{v1_dot_v2=}')

    m1 = Tensor(2, 2)
    m1.set(0, 0, 5)
    m1.set(0, 1, 6)
    m1.set(1, 0, 7)
    m1.set(1, 1, 8)
    print(f'{m1=}')

    m2 = Tensor(2, 2)
    m2.set(0, 0, 9)
    m2.set(0, 1, 10)
    m2.set(1, 0, 11)
    m2.set(1, 1, 12)
    print(f'{m2=}')

    m3 = matrix_multiply(m1, m2)
    print(f'{m3=}')

    m4 = matrix_multiply(m1, t1)
    print(f'{m4=}')

    m5 = matrix_multiply(v1, t1.transpose())
    print(f'{m5=}')

    m6 = matrix_multiply(t1, v1.transpose())
    print(f'{m6=}')


if __name__ == '__main__':
    test_tensors()
