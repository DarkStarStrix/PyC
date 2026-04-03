class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is None:
            self.data = [[0] * cols for _ in range(rows)]
        else:
            if len(data) != rows or any(len(row) != cols for row in data):
                raise ValueError("Data dimensions don't match specified size")
            self.data = [row[:] for row in data]

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

    def __getitem__(self, key):
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key, value):
        i, j = key
        self.data[i][j] = value


class MatrixMultiplier:
    @staticmethod
    def multiply(matrix_a, matrix_b):
        if matrix_a.cols != matrix_b.rows:
            raise ValueError("Matrix dimensions incompatible for multiplication")

        result = Matrix(matrix_a.rows, matrix_b.cols)

        for i in range(matrix_a.rows):
            for j in range(matrix_b.cols):
                sum_val = 0
                for k in range(matrix_a.cols):
                    sum_val += matrix_a[i, k] * matrix_b[k, j]
                result[i, j] = sum_val

        return result


def main():
    # Example usage
    a = Matrix(2, 2, [[1, 2], [3, 4]])
    b = Matrix(2, 2, [[5, 6], [7, 8]])

    print("Matrix A:")
    print(a)
    print("\nMatrix B:")
    print(b)

    multiplier = MatrixMultiplier()
    c = multiplier.multiply(a, b)

    print("\nResult C = A × B:")
    print(c)


if __name__ == "__main__":
    main()
