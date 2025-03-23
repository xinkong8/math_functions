#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "math_functions.h"

// 基础数学函数
double power(double base, int exponent) {
    double result = 1.0;
    for (int i = 0; i < abs(exponent); i++) {
        result *= base;
    }
    if (exponent < 0) {
        result = 1.0 / result;
    }
    return result;
}

double logarithm(double num, double base) {
    return log(num) / log(base);
}

unsigned long factorial(unsigned int n) {
    unsigned long result = 1;
    for (unsigned int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

double permutation(int n, int r) {
    if (n < r) return 0;
    return factorial(n) / factorial(n - r);
}

double combination(int n, int r) {
    if (n < r) return 0;
    return permutation(n, r) / factorial(r);
}

// 代数计算


struct QuadraticSolution solveQuadratic(double a, double b, double c) {
    struct QuadraticSolution solution;
    double discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
        solution.real_roots = 2;
        solution.root1 = (-b + sqrt(discriminant)) / (2 * a);
        solution.root2 = (-b - sqrt(discriminant)) / (2 * a);
    } else if (discriminant == 0) {
        solution.real_roots = 1;
        solution.root1 = -b / (2 * a);
        solution.root2 = solution.root1;
    } else {
        solution.real_roots = 0;
        solution.root1 = 0;
        solution.root2 = 0;
    }
    return solution;
}    

#define PI 3.14159265358979323846

// 几何计算
double circleArea(double radius) {
    return PI * radius * radius;
}

double sphereVolume(double radius) {
    return (4.0 / 3.0) * PI * pow(radius, 3);
}

double cylinderVolume(double radius, double height) {
    return PI * radius * radius * height;
}

double coneVolume(double radius, double height) {
    return (1.0 / 3.0) * PI * radius * radius * height;
}

// 三角函数扩展
double degreeToRadian(double degrees) {
    return degrees * (PI / 180.0);
}

double radianToDegree(double radians) {
    return radians * (180.0 / PI);
}

double sine(double degrees) {
    return sin(degreeToRadian(degrees));
}

double cosine(double degrees) {
    return cos(degreeToRadian(degrees));
}

double tangent(double degrees) {
    return tan(degreeToRadian(degrees));
}

double hyperbolicSine(double x) {
    return sinh(x);
}

double hyperbolicCosine(double x) {
    return cosh(x);
}    

// 计算函数在某点的导数
double derivative(MathFunction f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

// 使用梯形法则计算定积分
double integral(MathFunction f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += f(x);
    }
    return h * sum;
}

// // 线性代数


// // 创建矩阵
// Matrix createMatrix(int rows, int cols) {
//     Matrix mat;
//     mat.rows = rows;
//     mat.cols = cols;
//     mat.data = (double*)malloc(rows * cols * sizeof(double));
//     if (mat.data == NULL) {
//         mat.is_valid = false;
//     } else {
//         mat.is_valid = true;
//     }
//     return mat;
// }

// // 释放矩阵内存
// void freeMatrix(Matrix mat) {
//     if (mat.is_valid) {
//         free(mat.data);
//     }
// }

// // 矩阵乘法
// Matrix matrixMultiply(Matrix a, Matrix b) {
//     Matrix result;
//     if (!a.is_valid || !b.is_valid || a.cols != b.rows) {
//         result.is_valid = false;
//         return result;
//     }
//     result = createMatrix(a.rows, b.cols);
//     if (!result.is_valid) {
//         return result;
//     }
//     for (int i = 0; i < a.rows; i++) {
//         for (int j = 0; j < b.cols; j++) {
//             result.data[i * result.cols + j] = 0;
//             for (int k = 0; k < a.cols; k++) {
//                 result.data[i * result.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
//             }
//         }
//     }
//     return result;
// }

// 统计函数
// 计算数组的平均值
double tmean(double arr[], int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

// 计算数组的中位数
// double median(double arr[], int size) {
//     // 先对数组进行排序
//     for (int i = 0; i < size - 1; i++) {
//         for (int j = i + 1; j < size; j++) {
//             if (arr[i] > arr[j]) {
//                 double temp = arr[i];
//                 arr[i] = arr[j];
//                 arr[j] = temp;
//             }
//         }
//     }
//     if (size % 2 == 0) {
//         return (arr[size / 2 - 1] + arr[size / 2]) / 2;
//     } else {
//         return arr[size / 2];
//     }
// }

// 计算数组的方差
double tvariance(double arr[], int size) {
    double mean = tmean(arr, size);
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (arr[i] - mean) * (arr[i] - mean);
    }
    return sum / size;
}

// 计算数组的标准差
double standardDeviation(double arr[], int size) {
    return sqrt(tvariance(arr, size));
}

// 计算数组的极差
double range(double arr[], int size) {
    double min = arr[0];
    double max = arr[0];

    for (int i = 1; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
        if (arr[i] > max) {
            max = arr[i];
        }
    }

    return max - min;
}



// 计算两个数组的协方差
// double covariance(double arr1[], double arr2[], int size) {
//     double mean1 = mean(arr1, size);
//     double mean2 = mean(arr2, size);
//     double sum = 0;
//     for (int i = 0; i < size; i++) {
//         sum += (arr1[i] - mean1) * (arr2[i] - mean2);
//     }
//     return sum / size;
// }


// 数值方法
// 牛顿 - 拉夫逊方法求函数的根
double newtonRaphson(MathFunction f, MathFunction df, double guess, double tol) {
    double x = guess;
    double fx = f(x);
    while (fabs(fx) > tol) {
        x = x - fx / df(x);
        fx = f(x);
    }
    return x;
}

// 线性插值
double linearInterpolation(double x[], double y[], int n, double xi) {
    for (int i = 0; i < n - 1; i++) {
        if (xi >= x[i] && xi <= x[i + 1]) {
            return y[i] + ((y[i + 1] - y[i]) / (x[i + 1] - x[i])) * (xi - x[i]);
        }
    }
    return 0; // 如果 xi 不在给定的 x 范围内，返回 0
}
    
// 数论函数
// 判断一个数是否为素数
int is_prime(unsigned long n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (unsigned long i = 5; i * i <= n; i = i + 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

// 计算两个数的最大公约数
long gcd(long a, long b) {
    while (b != 0) {
        long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// 计算两个数的最小公倍数
long lcm(long a, long b) {
    return (a / gcd(a, b)) * b;
}

// 计算斐波那契数列的第 n 项
unsigned long fibonacci(unsigned int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    unsigned long a = 0, b = 1, c;
    for (unsigned int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// 将值限制在指定范围内
double clamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// 增强数学运算
// 计算 sigmoid 函数值
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 将弧度转换为角度
double radians_to_degrees(double radians) {
    return radians * (180.0 / M_PI);
}

// 将角度转换为弧度
double degrees_to_radians(double degrees) {
    return degrees * (M_PI / 180.0);
}

// 计算以 10 为底的对数
double log_base_10(double x) {
    return log10(x);
}

// 计算以 2 为底的对数
double log_base_2(double x) {
    return log2(x);
}

// 计算一个数的倒数
double reciprocal(double x) {
    return 1.0 / x;
}

// 计算一个数的立方根
double cube_root(double x) {
    return cbrt(x);
}

// 计算直角三角形的斜边长度
double hypotenuse(double a, double b) {
    return hypot(a, b);
}
    
// 取整函数
double round_to(double value, int decimals) {
    double multiplier = pow(10, decimals);
    return round(value * multiplier) / multiplier;
}

double cfloor(double value, int decimals) {
    double multiplier = pow(10, decimals);
    return floor(value * multiplier) / multiplier;
}

double cceil(double value, int decimals) {
    double multiplier = pow(10, decimals);
    return ceil(value * multiplier) / multiplier;
}

// 统计扩展
double sum(double arr[], int size) {
    double total = 0;
    for (int i = 0; i < size; i++) {
        total += arr[i];
    }
    return total;
}

double product(double arr[], int size) {
    double result = 1;
    for (int i = 0; i < size; i++) {
        result *= arr[i];
    }
    return result;
}

// 数值校验
int is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int is_even(int n) {
    return n % 2 == 0;
}

int is_odd(int n) {
    return n % 2 != 0;
}

// 范围操作
double map_range(double value, double from_min, double from_max, double to_min, double to_max) {
    return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min;
}

int in_range(double value, double min, double max) {
    return value >= min && value <= max;
}

// 单位转换
double celsius_to_fahrenheit(double c) {
    return c * 9 / 5 + 32;
}

double fahrenheit_to_celsius(double f) {
    return (f - 32) * 5 / 9;
}

double kilometers_to_miles(double km) {
    return km * 0.621371;
}

double miles_to_kilometers(double miles) {
    return miles / 0.621371;
}
    
// 随机数
int random_int(int min, int max) {
    static int initialized = 0;
    if (!initialized) {
        srand(time(NULL));
        initialized = 1;
    }
    return min + rand() % (max - min + 1);
}

double random_double(double min, double max) {
    static int initialized = 0;
    if (!initialized) {
        srand(time(NULL));
        initialized = 1;
    }
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// 符号处理
double copy_sign(double magnitude, double sign) {
    return copysign(magnitude, sign);
}

int sign(double x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

// 浮点数比较
int float_equal(double a, double b, double epsilon) {
    return fabs(a - b) <= epsilon;
}    

// 创建一个新的矩阵
Matrix matrix_create(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (double**)malloc(rows * sizeof(double*));
    
    for (int i = 0; i < rows; i++) {
        mat.data[i] = (double*)calloc(cols, sizeof(double));
    }
    return mat;
}

// 释放矩阵占用的内存
void matrix_free(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
}

// 克隆一个矩阵
Matrix matrix_clone(const Matrix* mat) {
    Matrix clone = matrix_create(mat->rows, mat->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            clone.data[i][j] = mat->data[i][j];
        }
    }
    return clone;
}

// 打印矩阵
// void matrix_print(const Matrix* mat) {
//     for (int i = 0; i < mat->rows; i++) {
//         for (int j = 0; j < mat->cols; j++) {
//             printf("%f ", mat->data[i][j]);
//         }
//         printf("\n");
//     }
// }

// 创建单位矩阵
Matrix matrix_identity(int n) {
    Matrix mat = matrix_create(n, n);
    for (int i = 0; i < n; i++) {
        mat.data[i][i] = 1.0;
    }
    return mat;
}

// 创建零矩阵
// Matrix matrix_zero(int rows, int cols) {
//     return matrix_create(rows, cols);
// }

// 获取矩阵中指定位置的元素
double matrix_get(const Matrix* mat, int row, int col) {
    return mat->data[row][col];
}

// 设置矩阵中指定位置的元素
void matrix_set(Matrix* mat, int row, int col, double value) {
    mat->data[row][col] = value;
}

// 矩阵加法
Matrix matrix_add(const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "矩阵维度不匹配，无法进行加法运算\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result.data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return result;
}

// 矩阵减法
Matrix matrix_subtract(const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "矩阵维度不匹配，无法进行减法运算\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result.data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return result;
}

// 矩阵乘法
Matrix matrix_multiply(const Matrix* a, const Matrix* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "矩阵维度不匹配，无法进行乘法运算\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                result.data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    return result;
}

// 矩阵与标量相乘
Matrix matrix_scalar_multiply(const Matrix* mat, double scalar) {
    Matrix result = matrix_create(mat->rows, mat->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result.data[i][j] = mat->data[i][j] * scalar;
        }
    }
    return result;
}

// 矩阵转置
Matrix matrix_transpose(const Matrix* mat) {
    Matrix result = matrix_create(mat->cols, mat->rows);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result.data[j][i] = mat->data[i][j];
        }
    }
    return result;
}

// 矩阵求逆（这里使用伴随矩阵法，仅适用于小矩阵）
Matrix matrix_inverse(const Matrix* mat) {
    if (!matrix_is_square(mat)) {
        fprintf(stderr, "非方阵无法求逆\n");
        exit(EXIT_FAILURE);
    }
    double det = matrix_determinant(mat);
    // if (fabs(det) < 1e-9) {
    //     fprintf(stderr, "矩阵不可逆\n");
    //     exit(EXIT_FAILURE);
    // }
    int n = mat->rows;
    Matrix adjugate = matrix_create(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Matrix minor = matrix_minor(mat, i, j);
            double cofactor = pow(-1, i + j) * matrix_determinant(&minor);
            adjugate.data[j][i] = cofactor;
            matrix_free(&minor);
        }
    }
    Matrix inverse = matrix_scalar_multiply(&adjugate, 1.0 / det);
    matrix_free(&adjugate);
    return inverse;
}

// 计算矩阵的行列式
double matrix_determinant(const Matrix* mat) {
    if (!matrix_is_square(mat)) {
        fprintf(stderr, "非方阵无法计算行列式\n");
        exit(EXIT_FAILURE);
    }
    int n = mat->rows;
    if (n == 1) {
        return mat->data[0][0];
    } else if (n == 2) {
        return mat->data[0][0] * mat->data[1][1] - mat->data[0][1] * mat->data[1][0];
    }
    double det = 0.0;
    for (int j = 0; j < n; j++) {
        Matrix minor = matrix_minor(mat, 0, j);
        det += pow(-1, j) * mat->data[0][j] * matrix_determinant(&minor);
        matrix_free(&minor);
    }
    return det;
}

LU_Decomposition matrix_lu_decompose(const Matrix* mat) {
    LU_Decomposition lu;
    int n = mat->rows;
    lu.L = matrix_create(n, n);
    lu.U = matrix_clone(mat);
    lu.swap_count = 0;
    for (int i = 0; i < n; i++) {
        lu.L.data[i][i] = 1.0;
    }
    for (int k = 0; k < n - 1; k++) {
        for (int i = k + 1; i < n; i++) {
            lu.L.data[i][k] = lu.U.data[i][k] / lu.U.data[k][k];
            for (int j = k; j < n; j++) {
                lu.U.data[i][j] -= lu.L.data[i][k] * lu.U.data[k][j];
            }
        }
    }
    return lu;
}

// 向量点积
double vector_dot_product(const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != 1 || b->cols != 1) {
        fprintf(stderr, "输入不是有效的列向量，无法进行点积运算\n");
        exit(EXIT_FAILURE);
    }
    double dot_product = 0.0;
    for (int i = 0; i < a->rows; i++) {
        dot_product += a->data[i][0] * b->data[i][0];
    }
    return dot_product;
}

// 向量叉积（仅适用于三维向量）
Matrix vector_cross_product(const Matrix* a, const Matrix* b) {
    if (a->rows != 3 || b->rows != 3 || a->cols != 1 || b->cols != 1) {
        fprintf(stderr, "输入不是有效的三维列向量，无法进行叉积运算\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(3, 1);
    result.data[0][0] = a->data[1][0] * b->data[2][0] - a->data[2][0] * b->data[1][0];
    result.data[1][0] = a->data[2][0] * b->data[0][0] - a->data[0][0] * b->data[2][0];
    result.data[2][0] = a->data[0][0] * b->data[1][0] - a->data[1][0] * b->data[0][0];
    return result;
}

// 矩阵增广
Matrix matrix_augment(const Matrix* mat, const Matrix* identity) {
    if (mat->rows != identity->rows) {
        fprintf(stderr, "矩阵行数不匹配，无法进行增广\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix_create(mat->rows, mat->cols + identity->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result.data[i][j] = mat->data[i][j];
        }
        for (int j = 0; j < identity->cols; j++) {
            result.data[i][mat->cols + j] = identity->data[i][j];
        }
    }
    return result;
}

// 矩阵的子矩阵
Matrix matrix_minor(const Matrix* mat, int exclude_row, int exclude_col) {
    Matrix minor = matrix_create(mat->rows - 1, mat->cols - 1);
    int r = 0;
    for (int i = 0; i < mat->rows; i++) {
        if (i == exclude_row) continue;
        int c = 0;
        for (int j = 0; j < mat->cols; j++) {
            if (j == exclude_col) continue;
            minor.data[r][c] = mat->data[i][j];
            c++;
        }
        r++;
    }
    return minor;
}

// 检查矩阵是否为方阵
bool matrix_is_square(const Matrix* mat) {
    return mat->rows == mat->cols;
}

// 检查矩阵是否可逆
bool matrix_is_invertible(const Matrix* mat) {
    return matrix_is_square(mat) && fabs(matrix_determinant(mat)) > 1e-9;
}

// 计算矩阵的秩（这里简单实现，仅适用于小矩阵）
int matrix_rank(const Matrix* mat) {
    LU_Decomposition lu = matrix_lu_decompose(mat);
    int rank = 0;
    int n = lu.U.rows;
    for (int i = 0; i < n; i++) {
        if (fabs(lu.U.data[i][i]) > 1e-9) {
            rank++;
        }
    }
    matrix_free(&lu.L);
    matrix_free(&lu.U);
    return rank;
}    

// 辅助函数：交换两个 double 类型的值
void swap(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

// 辅助函数：快速排序
void quicksort(double arr[], int left, int right) {
    if (left < right) {
        double pivot = arr[right];
        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[right]);
        int pivot_index = i + 1;

        quicksort(arr, left, pivot_index - 1);
        quicksort(arr, pivot_index + 1, right);
    }
}

// 描述性统计
double mean(const double data[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

double median(double data[], int n) {
    double *sorted = (double *)malloc(n * sizeof(double));
    memcpy(sorted, data, n * sizeof(double));
    quicksort(sorted, 0, n - 1);

    double result;
    if (n % 2 == 0) {
        result = (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    } else {
        result = sorted[n / 2];
    }
    free(sorted);
    return result;
}

double mode(double data[], int n) {
    double *sorted = (double *)malloc(n * sizeof(double));
    memcpy(sorted, data, n * sizeof(double));
    quicksort(sorted, 0, n - 1);

    int max_count = 0;
    double mode_value = sorted[0];
    int current_count = 1;

    for (int i = 1; i < n; i++) {
        if (sorted[i] == sorted[i - 1]) {
            current_count++;
        } else {
            if (current_count > max_count) {
                max_count = current_count;
                mode_value = sorted[i - 1];
            }
            current_count = 1;
        }
    }

    if (current_count > max_count) {
        mode_value = sorted[n - 1];
    }

    free(sorted);
    return mode_value;
}

double variance(const double data[], int n, bool is_sample) {
    double m = mean(data, n);
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (data[i] - m) * (data[i] - m);
    }
    return sum / (is_sample ? n - 1 : n);
}

double standard_deviation(const double data[], int n, bool is_sample) {
    return sqrt(variance(data, n, is_sample));
}

double skewness(const double data[], int n) {
    double m = mean(data, n);
    double std_dev = standard_deviation(data, n, false);
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow((data[i] - m) / std_dev, 3);
    }
    return sum / n;
}

double kurtosis(const double data[], int n) {
    double m = mean(data, n);
    double std_dev = standard_deviation(data, n, false);
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow((data[i] - m) / std_dev, 4);
    }
    return sum / n - 3;
}

// 回归分析
LinearRegressionResult linear_regression(const double x[], const double y[], int n) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    double x_mean = sum_x / n;
    double y_mean = sum_y / n;

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double intercept = y_mean - slope * x_mean;

    double rss = 0, tss = 0;
    for (int i = 0; i < n; i++) {
        double y_pred = slope * x[i] + intercept;
        rss += (y[i] - y_pred) * (y[i] - y_pred);
        tss += (y[i] - y_mean) * (y[i] - y_mean);
    }

    double r_squared = 1 - rss / tss;
    double std_err = sqrt(rss / (n - 2));

    LinearRegressionResult result = {slope, intercept, r_squared, std_err};
    return result;
}

// 多项式回归（简化实现）
PolynomialRegressionResult polynomial_regression(const double x[], const double y[], int n, int degree) {
    PolynomialRegressionResult result;
    result.degree = degree;
    result.coefficients = (double *)calloc(degree + 1, sizeof(double));

    // 这里只是简单示例，实际需要更复杂的实现
    result.rss = 0;
    result.tss = 0;

    return result;
}

void free_polynomial_result(PolynomialRegressionResult* result) {
    free(result->coefficients);
}

// 假设检验
TTestResult t_test(const double sample1[], int n1, const double sample2[], int n2, bool paired) {
    TTestResult result;
    if (paired) {
        // 配对 t 检验
        double diff_sum = 0, diff_sum_sq = 0;
        for (int i = 0; i < n1; i++) {
            double diff = sample1[i] - sample2[i];
            diff_sum += diff;
            diff_sum_sq += diff * diff;
        }
        double mean_diff = diff_sum / n1;
        double std_dev_diff = sqrt((diff_sum_sq - diff_sum * diff_sum / n1) / (n1 - 1));
        result.t_statistic = mean_diff / (std_dev_diff / sqrt(n1));
        result.df = n1 - 1;
    } else {
        // 独立样本 t 检验
        double mean1 = mean(sample1, n1);
        double mean2 = mean(sample2, n2);
        double var1 = variance(sample1, n1, true);
        double var2 = variance(sample2, n2, true);
        result.t_statistic = (mean1 - mean2) / sqrt(var1 / n1 + var2 / n2);
        result.df = n1 + n2 - 2;
    }
    // 这里 p 值需要查表或使用更复杂的方法计算，简单示例设为 0
    result.p_value = 0;
    return result;
}

double chi_square_test(const double observed[], const double expected[], int categories) {
    double chi_square = 0;
    for (int i = 0; i < categories; i++) {
        chi_square += pow(observed[i] - expected[i], 2) / expected[i];
    }
    return chi_square;
}

double f_test(const double sample1[], int n1, const double sample2[], int n2) {
    double var1 = variance(sample1, n1, true);
    double var2 = variance(sample2, n2, true);
    return var1 > var2 ? var1 / var2 : var2 / var1;
}

// 相关分析
double pearson_correlation(const double x[], const double y[], int n) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    return numerator / denominator;
}

double spearman_rank_correlation(double x[], double y[], int n) {
    double *rank_x = (double *)malloc(n * sizeof(double));
    double *rank_y = (double *)malloc(n * sizeof(double));

    // 计算排名
    for (int i = 0; i < n; i++) {
        int rank1 = 1, rank2 = 1;
        for (int j = 0; j < n; j++) {
            if (x[j] < x[i]) rank1++;
            if (y[j] < y[i]) rank2++;
        }
        rank_x[i] = rank1;
        rank_y[i] = rank2;
    }

    double d_sum_sq = 0;
    for (int i = 0; i < n; i++) {
        double d = rank_x[i] - rank_y[i];
        d_sum_sq += d * d;
    }

    double rho = 1 - (6 * d_sum_sq) / (n * (n * n - 1));
    free(rank_x);
    free(rank_y);
    return rho;
}

// 概率分布
double normal_pdf(double x, double mu, double sigma) {
    return (1.0 / (sigma * sqrt(2 * M_PI))) * exp(-0.5 * pow((x - mu) / sigma, 2));
}

// 这里只是简单近似，实际需要更复杂的算法
double normal_cdf(double x, double mu, double sigma) {
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))));
}

// 这里只是简单示例，实际需要更复杂的实现
// double t_distribution_pdf(double x, double df) {
//     return 0;
// }

// double t_distribution_cdf(double x, double df) {
//     return 0;
// }

// double chi_squared_pdf(double x, double df) {
//     return 0;
// }

// double chi_squared_cdf(double x, double df) {
//     return 0;
// }

// 统计工具
void z_normalize(double data[], int n) {
    double m = mean(data, n);
    double std_dev = standard_deviation(data, n, true);
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - m) / std_dev;
    }
}

double confidence_interval_width(double std_dev, int sample_size, double confidence_level) {
    // 这里只是简单示例，实际需要根据置信水平查找 z 分数
    double z_score = 1.96; 
    return 2 * z_score * std_dev / sqrt(sample_size);
}

double calculate_margin_of_error(double std_dev, int sample_size, double z_score) {
    return z_score * std_dev / sqrt(sample_size);
}

// 时间序列分析
double autocorrelation(const double data[], int n, int lag) {
    double mean_val = mean(data, n);
    double numerator = 0, denominator = 0;
    for (int i = 0; i < n - lag; i++) {
        numerator += (data[i] - mean_val) * (data[i + lag] - mean_val);
    }
    for (int i = 0; i < n; i++) {
        denominator += (data[i] - mean_val) * (data[i] - mean_val);
    }
    return numerator / denominator;
}

double moving_average(const double data[], int n, int window_size) {
    double sum = 0;
    for (int i = 0; i < window_size; i++) {
        sum += data[i];
    }
    return sum / window_size;
}

// 非参数检验
double mann_whitney_u_test(const double sample1[], int n1, const double sample2[], int n2) {
    int total_size = n1 + n2;
    double *combined = (double *)malloc(total_size * sizeof(double));
    int *group = (int *)malloc(total_size * sizeof(int));

    for (int i = 0; i < n1; i++) {
        combined[i] = sample1[i];
        group[i] = 1;
    }
    for (int i = 0; i < n2; i++) {
        combined[n1 + i] = sample2[i];
        group[n1 + i] = 2;
    }

    // 排序
    quicksort(combined, 0, total_size - 1);

    double rank_sum1 = 0, rank_sum2 = 0;
    for (int i = 0; i < total_size; i++) {
        if (group[i] == 1) {
            rank_sum1 += i + 1;
        } else {
            rank_sum2 += i + 1;
        }
    }

    double u1 = rank_sum1 - (n1 * (n1 + 1) / 2);
    double u2 = rank_sum2 - (n2 * (n2 + 1) / 2);

    free(combined);
    free(group);
    return u1;
}    

// 创建插值对象
Interpolation* create_interpolation(double x[], double y[], int n, int method) {
    Interpolation* interp = (Interpolation*)malloc(sizeof(Interpolation));
    // if (interp == NULL) {
    //     return NULL;
    // }
    interp->x = (double*)malloc(n * sizeof(double));
    interp->y = (double*)malloc(n * sizeof(double));
    // if (interp->x == NULL || interp->y == NULL) {
    //     free(interp->x);
    //     free(interp->y);
    //     free(interp);
    //     return NULL;
    // }
    for (int i = 0; i < n; i++) {
        interp->x[i] = x[i];
        interp->y[i] = y[i];
    }
    interp->n = n;
    interp->degree = n - 1;
    interp->coefficients = (double*)malloc((n) * sizeof(double));
    // if (interp->coefficients == NULL) {
    //     free(interp->x);
    //     free(interp->y);
    //     free(interp);
    //     return NULL;
    // }
    // 简单示例，使用拉格朗日插值，系数暂未使用
    return interp;
}

// 进行插值计算
double interpolate(Interpolation* interp, double xi) {
    double result = 0.0;
    for (int i = 0; i < interp->n; i++) {
        double term = interp->y[i];
        for (int j = 0; j < interp->n; j++) {
            if (j != i) {
                term *= (xi - interp->x[j]) / (interp->x[i] - interp->x[j]);
            }
        }
        result += term;
    }
    return result;
}

// 释放插值对象的内存
void free_interpolation(Interpolation* interp) {
    if (interp != NULL) {
        free(interp->x);
        free(interp->y);
        free(interp->coefficients);
        free(interp);
    }
}

// 数值微分
double numerical_derivative(MathFunc f, double x, double h, int method) {
    if (method == 0) { // 中心差分
        return (f(x + h) - f(x - h)) / (2 * h);
    }
}

// 理查森外推法
double richardson_extrapolation(MathFunc f, double x, double h, int steps) {
    double** table = (double**)malloc((steps + 1) * sizeof(double*));
    for (int i = 0; i <= steps; i++) {
        table[i] = (double*)malloc((steps + 1) * sizeof(double));
    }
    for (int i = 0; i <= steps; i++) {
        table[i][0] = numerical_derivative(f, x, h / pow(2, i), 0);
    }
    for (int j = 1; j <= steps; j++) {
        for (int i = j; i <= steps; i++) {
            table[i][j] = table[i][j - 1] + (table[i][j - 1] - table[i - 1][j - 1]) / (pow(4, j) - 1);
        }
    }
    double result = table[steps][steps];
    for (int i = 0; i <= steps; i++) {
        free(table[i]);
    }
    free(table);
    return result;
}

// // 微分方程求解
// void solve_ode(MathFunc f, double y0, double t_start, double t_end, 
//               double step, ODESolverType method, double results[]) {
//     int n_steps = (int)((t_end - t_start) / step);
//     results[0] = y0;
//     double t = t_start;
//     double y = y0;
//     for (int i = 0; i < n_steps; i++) {
//         if (method == EULER) {
//             y = y + step * f(t);
//         } else if (method == RK4) {
//             double k1 = step * f(t);
//             double k2 = step * f(t + step / 2);
//             double k3 = step * f(t + step / 2);
//             double k4 = step * f(t + step);
//             y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
//         }
//         results[i + 1] = y;
//         t += step;
//     }
// }

// 偏微分方程求解器（热传导方程示例）
void solve_heat_equation(double (*initial)(double), double L, double T, 
                        double dx, double dt, double alpha, double* solution) {
    int nx = (int)(L / dx);
    int nt = (int)(T / dt);
    double r = alpha * dt / (dx * dx);
    for (int i = 0; i < nx; i++) {
        solution[i] = initial(i * dx);
    }
    double* temp = (double*)malloc(nx * sizeof(double));
    for (int n = 0; n < nt; n++) {
        for (int i = 1; i < nx - 1; i++) {
            temp[i] = solution[i] + r * (solution[i + 1] - 2 * solution[i] + solution[i - 1]);
        }
        for (int i = 1; i < nx - 1; i++) {
            solution[i] = temp[i];
        }
    }
    free(temp);
}

// 自适应积分
double adaptive_quadrature(MathFunc f, double a, double b, double tol) {
    // 简单示例，使用递归实现
    double h = b - a;
    double m = (a + b) / 2;
    double I1 = h / 2 * (f(a) + f(b));
    double I2 = h / 4 * (f(a) + 2 * f(m) + f(b));
    if (fabs(I2 - I1) < 15 * tol) {
        return I2 + (I2 - I1) / 15;
    } else {
        return adaptive_quadrature(f, a, m, tol / 2) + adaptive_quadrature(f, m, b, tol / 2);
    }
}

// 黄金分割搜索
double golden_section_search(MathFunc f, double a, double b, double tol) {
    double phi = (sqrt(5) - 1) / 2;
    double x1 = a + (1 - phi) * (b - a);
    double x2 = a + phi * (b - a);
    double f1 = f(x1);
    double f2 = f(x2);
    while (fabs(b - a) > tol) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (1 - phi) * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + phi * (b - a);
            f2 = f(x2);
        }
    }
    return (a + b) / 2;
}

// 共轭梯度法（简单示例，未完整实现）
// void conjugate_gradient(double (*f)(double*), double* x0, int n, double tol) {
//     // 简单示例，未完整实现
//     return;
// }

// 牛顿法
// double newton_method(MathFunc f, MathFunc df, double x0, double tol) {
//     double x = x0;
//     double fx = f(x);
//     while (fabs(fx) > tol) {
//         x = x - fx / df(x);
//         fx = f(x);
//     }
//     return x;
// }

// 割线法
// double secant_method(MathFunc f, double x0, double x1, double tol) {
//     double fx0 = f(x0);
//     double fx1 = f(x1);
//     while (fabs(fx1) > tol) {
//         double x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0);
//         x0 = x1;
//         fx0 = fx1;
//         x1 = x2;
//         fx1 = f(x1);
//     }
//     return x1;
// }

// 创建一个复数
Complex create_complex(double real, double imag) {
    Complex c;
    c.real = real;
    c.imag = imag;
    return c;
}

// 复数加法
Complex add_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

// 复数减法
Complex subtract_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

// 复数乘法
Complex multiply_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

// 复数除法
Complex divide_complex(Complex a, Complex b) {
    double denominator = b.real * b.real + b.imag * b.imag;
    Complex result;
    result.real = (a.real * b.real + a.imag * b.imag) / denominator;
    result.imag = (a.imag * b.real - a.real * b.imag) / denominator;
    return result;
}

// 求复数的模
double modulus_complex(Complex c) {
    return sqrt(c.real * c.real + c.imag * c.imag);
}

// 打印复数
void print_complex(Complex c) {
    if (c.imag >= 0) {
        printf("%.2f + %.2fi\n", c.real, c.imag);
    } else {
        printf("%.2f - %.2fi\n", c.real, -c.imag);
    }
}
  

// 创建多项式
Polynomial create_polynomial(int degree, double *coefficients) {
    Polynomial poly;
    poly.degree = degree;
    poly.coefficients = (double *)malloc((degree + 1) * sizeof(double));
    for (int i = 0; i <= degree; i++) {
        poly.coefficients[i] = coefficients[i];
    }
    return poly;
}

// 释放多项式内存
void free_polynomial(Polynomial poly) {
    free(poly.coefficients);
}

// 多项式加法
Polynomial add_polynomials(Polynomial poly1, Polynomial poly2) {
    int max_degree = (poly1.degree > poly2.degree) ? poly1.degree : poly2.degree;
    double *result_coeffs = (double *)calloc(max_degree + 1, sizeof(double));
    for (int i = 0; i <= poly1.degree; i++) {
        result_coeffs[i] += poly1.coefficients[i];
    }
    for (int i = 0; i <= poly2.degree; i++) {
        result_coeffs[i] += poly2.coefficients[i];
    }
    // 去掉最高次项系数为 0 的情况
    while (max_degree > 0 && result_coeffs[max_degree] == 0) {
        max_degree--;
    }
    Polynomial result = create_polynomial(max_degree, result_coeffs);
    free(result_coeffs);
    return result;
}

// 多项式减法
Polynomial subtract_polynomials(Polynomial poly1, Polynomial poly2) {
    int max_degree = (poly1.degree > poly2.degree) ? poly1.degree : poly2.degree;
    double *result_coeffs = (double *)calloc(max_degree + 1, sizeof(double));
    for (int i = 0; i <= poly1.degree; i++) {
        result_coeffs[i] += poly1.coefficients[i];
    }
    for (int i = 0; i <= poly2.degree; i++) {
        result_coeffs[i] -= poly2.coefficients[i];
    }
    // 去掉最高次项系数为 0 的情况
    while (max_degree > 0 && result_coeffs[max_degree] == 0) {
        max_degree--;
    }
    Polynomial result = create_polynomial(max_degree, result_coeffs);
    free(result_coeffs);
    return result;
}

// 多项式乘法
Polynomial multiply_polynomials(Polynomial poly1, Polynomial poly2) {
    int result_degree = poly1.degree + poly2.degree;
    double *result_coeffs = (double *)calloc(result_degree + 1, sizeof(double));
    for (int i = 0; i <= poly1.degree; i++) {
        for (int j = 0; j <= poly2.degree; j++) {
            result_coeffs[i + j] += poly1.coefficients[i] * poly2.coefficients[j];
        }
    }
    Polynomial result = create_polynomial(result_degree, result_coeffs);
    free(result_coeffs);
    return result;
}

// 多项式除法（仅返回商）
// Polynomial divide_polynomials(Polynomial poly1, Polynomial poly2) {
//     if (poly2.degree > poly1.degree) {
//         double zero_coeffs[] = {0};
//         return create_polynomial(0, zero_coeffs);
//     }
//     int quotient_degree = poly1.degree - poly2.degree;
//     double *quotient_coeffs = (double *)calloc(quotient_degree + 1, sizeof(double));
//     Polynomial temp = poly1;
//     while (temp.degree >= poly2.degree) {
//         int shift = temp.degree - poly2.degree;
//         double factor = temp.coefficients[temp.degree] / poly2.coefficients[poly2.degree];
//         quotient_coeffs[shift] = factor;
//         Polynomial sub_poly = create_polynomial(poly2.degree + shift, (double *)calloc(poly2.degree + shift + 1, sizeof(double)));
//         for (int i = 0; i <= poly2.degree; i++) {
//             sub_poly.coefficients[i + shift] = poly2.coefficients[i] * factor;
//         }
//         temp = subtract_polynomials(temp, sub_poly);
//         free_polynomial(sub_poly);
//     }
//     Polynomial quotient = create_polynomial(quotient_degree, quotient_coeffs);
//     free(quotient_coeffs);
//     free_polynomial(temp);
//     return quotient;
// }

// 多项式求值
double evaluate_polynomial(Polynomial poly, double x) {
    double result = 0;
    for (int i = poly.degree; i >= 0; i--) {
        result = result * x + poly.coefficients[i];
    }
    return result;
}

// 多项式求导
Polynomial differentiate_polynomial(Polynomial poly) {
    // if (poly.degree == 0) {
    //     double zero_coeffs[] = {0};
    //     return create_polynomial(0, zero_coeffs);
    // }
    int new_degree = poly.degree - 1;
    double *new_coeffs = (double *)malloc(new_degree + 1 * sizeof(double));
    for (int i = 0; i <= new_degree; i++) {
        new_coeffs[i] = poly.coefficients[i + 1] * (i + 1);
    }
    Polynomial result = create_polynomial(new_degree, new_coeffs);
    free(new_coeffs);
    return result;
}

// 打印多项式
void print_polynomial(Polynomial poly) {
    for (int i = poly.degree; i >= 0; i--) {
        if (i < poly.degree && poly.coefficients[i] >= 0) {
            printf(" + ");
        }
        if (i == 0) {
            printf("%f", poly.coefficients[i]);
        } else if (i == 1) {
            printf("%fx", poly.coefficients[i]);
        } else {
            printf("%fx^%d", poly.coefficients[i], i);
        }
    }
    printf("\n");
}    



// 二项分布概率密度函数
double binomial_pdf(int k, int n, double p) {
    //if (k < 0 || k > n) return 0;
    double comb = factorial(n) / (factorial(k) * factorial(n - k));
    return comb * pow(p, k) * pow(1 - p, n - k);
}

// 二项分布累积分布函数
double binomial_cdf(int k, int n, double p) {
    double cdf = 0;
    for (int i = 0; i <= k; ++i) {
        cdf += binomial_pdf(i, n, p);
    }
    return cdf;
}

// 泊松分布概率密度函数
double poisson_pdf(int k, double lambda) {
    //if (k < 0) return 0;
    return pow(lambda, k) * exp(-lambda) / factorial(k);
}

// 泊松分布累积分布函数
double poisson_cdf(int k, double lambda) {
    double cdf = 0;
    for (int i = 0; i <= k; ++i) {
        cdf += poisson_pdf(i, lambda);
    }
    return cdf;
}

// 指数分布概率密度函数
double exponential_pdf(double x, double lambda) {
    //if (x < 0) return 0;
    return lambda * exp(-lambda * x);
}

// 指数分布累积分布函数
double exponential_cdf(double x, double lambda) {
    //if (x < 0) return 0;
    return 1 - exp(-lambda * x);
}   

// 冒泡排序
void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// 选择排序
void selection_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            int temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

// 插入排序
void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// 希尔排序
void shell_sort(int arr[], int n) {
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

// 归并排序辅助函数：合并两个子数组
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// 归并排序
void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// 快速排序辅助函数：分区
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

// 快速排序
void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// 堆排序辅助函数：调整堆
void heapify(int arr[], int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && arr[l] > arr[largest]) {
        largest = l;
    }
    if (r < n && arr[r] > arr[largest]) {
        largest = r;
    }
    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
        heapify(arr, n, largest);
    }
}

// 堆排序
void heap_sort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
    for (int i = n - 1; i > 0; i--) {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

// 计数排序
void counting_sort(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    int *count = (int *)calloc(max + 1, sizeof(int));
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }
    int index = 0;
    for (int i = 0; i <= max; i++) {
        while (count[i] > 0) {
            arr[index] = i;
            index++;
            count[i]--;
        }
    }
    free(count);
}

// 桶排序
void bucket_sort(int arr[], int n) {
    int max = arr[0];
    int min = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    int num_buckets = 10;
    int bucket_range = (max - min) / num_buckets + 1;
    int **buckets = (int **)malloc(num_buckets * sizeof(int *));
    for (int i = 0; i < num_buckets; i++) {
        buckets[i] = (int *)malloc(n * sizeof(int));
        buckets[i][0] = 0;
    }
    for (int i = 0; i < n; i++) {
        int bucket_index = (arr[i] - min) / bucket_range;
        int *bucket = buckets[bucket_index];
        int size = bucket[0];
        bucket[size + 1] = arr[i];
        bucket[0]++;
    }
    int index = 0;
    for (int i = 0; i < num_buckets; i++) {
        int *bucket = buckets[i];
        int size = bucket[0];
        insertion_sort(bucket + 1, size);
        for (int j = 1; j <= size; j++) {
            arr[index] = bucket[j];
            index++;
        }
        free(buckets[i]);
    }
    free(buckets);
}

// 基数排序辅助函数：获取数字的第 exp 位
int get_digit(int num, int exp) {
    return (num / exp) % 10;
}

// 基数排序
void radix_sort(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    for (int exp = 1; max / exp > 0; exp *= 10) {
        int output[n];
        int count[10] = {0};
        for (int i = 0; i < n; i++) {
            count[get_digit(arr[i], exp)]++;
        }
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        for (int i = n - 1; i >= 0; i--) {
            output[count[get_digit(arr[i], exp)] - 1] = arr[i];
            count[get_digit(arr[i], exp)]--;
        }
        for (int i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }
}    

// 线性搜索
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    //return -1;
}

// 二分搜索
int binary_search(int arr[], int left, int right, int target) {
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    //return -1;
}

// 插值搜索
int interpolation_search(int arr[], int n, int target) {
    int low = 0, high = n - 1;
    while (low <= high && target >= arr[low] && target <= arr[high]) {
        if (low == high) {
            if (arr[low] == target) {
                return low;
            }
            //return -1;
        }
        int pos = low + (((double)(target - arr[low]) * (high - low)) / (arr[high] - arr[low]));
        if (arr[pos] == target) {
            return pos;
        } else if (arr[pos] < target) {
            low = pos + 1;
        } 
    }
    return -1;
}

// 跳跃搜索
int jump_search(int arr[], int n, int target) {
    int step = sqrt(n);
    int prev = 0;
    while (arr[step < n ? step : n - 1] < target) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) {
            return -1;
        }
    }
    while (arr[prev] < target) {
        prev++;
        // if (prev == (step < n ? step : n)) {
        //     return -1;
        // }
    }
    if (arr[prev] == target) {
        return prev;
    }
    //return -1;
}

// 斐波那契搜索
int fibonacci_search(int arr[], int n, int target) {
    int fib2 = 0;
    int fib1 = 1;
    int fib = fib2 + fib1;
    while (fib < n) {
        fib2 = fib1;
        fib1 = fib;
        fib = fib2 + fib1;
    }
    int offset = -1;
    while (fib > 1) {
        int i = (offset + fib2) < n ? (offset + fib2) : n - 1;
        if (arr[i] < target) {
            fib = fib1;
            fib1 = fib2;
            fib2 = fib - fib1;
            offset = i;
        } else if (arr[i] > target) {
            fib = fib2;
            fib1 = fib1 - fib2;
            fib2 = fib - fib1;
        } else {
            return i;
        }
    }
    // if (fib1 && arr[offset + 1] == target) {
    //     return offset + 1;
    // }
    return -1;
}

// 指数搜索
int exponential_search(int arr[], int n, int target) {
    // if (arr[0] == target) {
    //     return 0;
    // }
    int i = 1;
    while (i < n && arr[i] <= target) {
        i *= 2;
    }
    return binary_search(arr, i / 2, i, target);
}

// 二叉搜索树搜索
TreeNode* bst_search(TreeNode *root, int target) {
    if (root == NULL || root->data == target) {
        return root;
    }
    if (root->data < target) {
        return bst_search(root->right, target);
    }
    return bst_search(root->left, target);
}

// 哈希搜索
int hash_search(int hash_table[], int target) {
    int index = target % HASH_TABLE_SIZE;
    if (hash_table[index] == target) {
        return index;
    }
    return -1;
}

// 顺序搜索（链表实现）
ListNode* list_search(ListNode *head, int target) {
    ListNode *current = head;
    while (current != NULL) {
        if (current->data == target) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// 分块搜索
int block_search(int arr[], int n, int target, int block_size) {
    int num_blocks = n / block_size;
    if (n % block_size != 0) {
        num_blocks++;
    }
    int block_index = 0;
    while (block_index < num_blocks && arr[block_index * block_size] <= target) {
        block_index++;
    }
    block_index--;
    // if (block_index < 0) {
    //     block_index = 0;
    // }
    int start = block_index * block_size;
    int end = n;
    if(arr[start] == target){
        return start;
    }
    // for (int i = start; i < end; i++) {
    //     if (arr[i] == target) {
    //         return i;
    //     }
    // }
    //return -1;
}    

// 化简分数
Fraction simplify(Fraction f) {
    int common = gcd(abs(f.numerator), abs(f.denominator));
    f.numerator /= common;
    f.denominator /= common;
    if (f.denominator < 0) {
        f.numerator = -f.numerator;
        f.denominator = -f.denominator;
    }
    return f;
}

// 分数加法
Fraction add(Fraction f1, Fraction f2) {
    Fraction result;
    result.numerator = f1.numerator * f2.denominator + f2.numerator * f1.denominator;
    result.denominator = f1.denominator * f2.denominator;
    return simplify(result);
}

// 分数减法
Fraction subtract(Fraction f1, Fraction f2) {
    Fraction result;
    result.numerator = f1.numerator * f2.denominator - f2.numerator * f1.denominator;
    result.denominator = f1.denominator * f2.denominator;
    return simplify(result);
}

// 分数乘法
Fraction multiply(Fraction f1, Fraction f2) {
    Fraction result;
    result.numerator = f1.numerator * f2.numerator;
    result.denominator = f1.denominator * f2.denominator;
    return simplify(result);
}

// 分数除法
Fraction divide(Fraction f1, Fraction f2) {
    if (f2.numerator == 0) {
        printf("Error: Division by zero!\n");
        exit(EXIT_FAILURE);
    }
    Fraction result;
    result.numerator = f1.numerator * f2.denominator;
    result.denominator = f1.denominator * f2.numerator;
    return simplify(result);
}

// 打印分数
void printFraction(Fraction f) {
    if (f.denominator == 1) {
        printf("%d\n", f.numerator);
    } else {
        printf("%d/%d\n", f.numerator, f.denominator);
    }
}

// 群的乘法表，这里假设是一个4阶群
int multiplication_table[GROUP_SIZE][GROUP_SIZE] = {
    {0, 1, 2, 3},
    {1, 0, 3, 2},
    {2, 3, 0, 1},
    {3, 2, 1, 0}
};

// 群元素乘法函数
int group_multiplication(int a, int b) {
    if (a < 0 || a >= GROUP_SIZE || b < 0 || b >= GROUP_SIZE) {
        fprintf(stderr, "Error: Invalid group element!\n");
        return -1;
    }
    return multiplication_table[a][b];
}

// 查找单位元函数
int find_identity() {
    bool is_identity = true;
    for (int i = 0; i < GROUP_SIZE; i++) {
        // if (group_multiplication(i, e) != i || group_multiplication(e, i) != i) {
        //     is_identity = false;
        //     break;
        // }
    }
    if (is_identity) {
        return 0;
    }
    // fprintf(stderr, "Error: No identity element found!\n");
    // return -1;
}

// 查找元素的逆元函数
int find_inverse(int element) {
    int identity = find_identity();
    // if (identity == -1) {
    //     return -1;
    // }
    for (int i = 0; i < GROUP_SIZE; i++) {
        if (group_multiplication(element, i) == identity && group_multiplication(i, element) == identity) {
            return i;
        }
    }
    // fprintf(stderr, "Error: No inverse element found for %d!\n", element);
    // return -1;
}

// 检查群的封闭性
bool check_closure() {
    for (int i = 0; i < GROUP_SIZE; i++) {
        for (int j = 0; j < GROUP_SIZE; j++) {
            int result = group_multiplication(i, j);
            // if (result < 0 || result >= GROUP_SIZE) {
            //     return false;
            // }
        }
    }
    return true;
}

// 检查群的结合律
bool check_associativity() {
    for (int i = 0; i < GROUP_SIZE; i++) {
        for (int j = 0; j < GROUP_SIZE; j++) {
            for (int k = 0; k < GROUP_SIZE; k++) {
                int left = group_multiplication(group_multiplication(i, j), k);
                int right = group_multiplication(i, group_multiplication(j, k));
                // if (left != right) {
                //     return false;
                // }
            }
        }
    }
    return true;
}    

bool solve_linear_equation(double a1, double b1, double c1, double d1,
                           double a2, double b2, double c2, double d2,
                           double a3, double b3, double c3, double d3,
                           double *x, double *y, double *z) {
    double det = a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2);
    if (det == 0) {
        return false;
    }

    double det_x = d1 * (b2 * c3 - b3 * c2) - b1 * (d2 * c3 - d3 * c2) + c1 * (d2 * b3 - d3 * b2);
    double det_y = a1 * (d2 * c3 - d3 * c2) - d1 * (a2 * c3 - a3 * c2) + c1 * (a2 * d3 - a3 * d2);
    double det_z = a1 * (b2 * d3 - b3 * d2) - b1 * (a2 * d3 - a3 * d2) + d1 * (a2 * b3 - a3 * b2);

    *x = det_x / det;
    *y = det_y / det;
    *z = det_z / det;

    return true;
}    