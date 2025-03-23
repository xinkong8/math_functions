#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

// 基础数学函数
double power(double base, int exponent);
double logarithm(double num, double base);
unsigned long factorial(unsigned int n);
double permutation(int n, int r);
double combination(int n, int r);

// 代数计算
struct QuadraticSolution {
    double root1;
    double root2;
    int real_roots; // 0: 无实根, 1: 单根, 2: 双根
};
struct QuadraticSolution solveQuadratic(double a, double b, double c);

// 几何计算
#define PI 3.14159265358979323846
double circleArea(double radius);
double sphereVolume(double radius);
double cylinderVolume(double radius, double height);
double coneVolume(double radius, double height);

// 三角函数扩展
double degreeToRadian(double degrees);
double radianToDegree(double radians);
double sine(double degrees);
double cosine(double degrees);
double tangent(double degrees);
double hyperbolicSine(double x);
double hyperbolicCosine(double x);

// 微积分相关
typedef double (*MathFunction)(double);
double derivative(MathFunction f, double x, double h);
double integral(MathFunction f, double a, double b, int n);

// // 线性代数
// typedef struct {
//     double* data;
//     int rows;
//     int cols;
//     bool is_valid;  // 用于错误状态标记
// } Matrix;


// Matrix createMatrix(int rows, int cols);
// void freeMatrix(Matrix mat);
// Matrix matrixMultiply(Matrix a, Matrix b);

// 统计函数
double tmean(double arr[], int size);
double median(double arr[], int size);
double tvariance(double arr[], int size);
double standardDeviation(double arr[], int size);
double range(double arr[], int size);
// 计算两个数组的协方差
//double covariance(double arr1[], double arr2[], int size);

// 数值方法
double newtonRaphson(MathFunction f, MathFunction df, double guess, double tol);
double linearInterpolation(double x[], double y[], int n, double xi);

// 数论函数
int is_prime(unsigned long n);
long gcd(long a, long b);
long lcm(long a, long b);
unsigned long fibonacci(unsigned int n);
double clamp(double value, double min, double max);

// 增强数学运算
double sigmoid(double x);
double radians_to_degrees(double radians);
double degrees_to_radians(double degrees);
double log_base_10(double x);
double log_base_2(double x);
double reciprocal(double x);
double cube_root(double x);
double hypotenuse(double a, double b);

// 取整函数
double round_to(double value, int decimals);
double cfloor(double value, int decimals);
double cceil(double value, int decimals);

// 统计扩展
double sum(double arr[], int size);
double product(double arr[], int size);

// 数值校验
int is_power_of_two(int n);
int is_even(int n);
int is_odd(int n);

// 范围操作
double map_range(double value, double from_min, double from_max, double to_min, double to_max);
int in_range(double value, double min, double max);

// 单位转换
double celsius_to_fahrenheit(double c);
double fahrenheit_to_celsius(double f);
double kilometers_to_miles(double km);
double miles_to_kilometers(double miles);

// 随机数
int random_int(int min, int max);
double random_double(double min, double max);

// 符号处理
double copy_sign(double magnitude, double sign);
int sign(double x);

// 浮点数比较
int float_equal(double a, double b, double epsilon);

// 定义 Matrix 结构体
typedef struct {
    int rows;
    int cols;
    double** data;
    bool is_valid;
} Matrix;

// 基础操作
Matrix matrix_create(int rows, int cols);
void matrix_free(Matrix* mat);
Matrix matrix_clone(const Matrix* mat);
//void matrix_print(const Matrix* mat);
Matrix matrix_identity(int n);
Matrix matrix_zero(int rows, int cols);

// 存取操作
double matrix_get(const Matrix* mat, int row, int col);
void matrix_set(Matrix* mat, int row, int col, double value);

// 矩阵运算
Matrix matrix_add(const Matrix* a, const Matrix* b);
Matrix matrix_subtract(const Matrix* a, const Matrix* b);
Matrix matrix_multiply(const Matrix* a, const Matrix* b);
Matrix matrix_scalar_multiply(const Matrix* mat, double scalar);
Matrix matrix_transpose(const Matrix* mat);
Matrix matrix_inverse(const Matrix* mat);
double matrix_determinant(const Matrix* mat);

// 矩阵分解
typedef struct {
    Matrix L;
    Matrix U;
    int swap_count;
} LU_Decomposition;
LU_Decomposition matrix_lu_decompose(const Matrix* mat);

// 向量运算
double vector_dot_product(const Matrix* a, const Matrix* b);
Matrix vector_cross_product(const Matrix* a, const Matrix* b);

// 特殊操作
Matrix matrix_augment(const Matrix* mat, const Matrix* identity);
Matrix matrix_minor(const Matrix* mat, int exclude_row, int exclude_col);

// 属性检查
bool matrix_is_square(const Matrix* mat);
bool matrix_is_invertible(const Matrix* mat);
int matrix_rank(const Matrix* mat);

// 数据结构定义
typedef struct {
    double* data;
    int size;
} Dataset;

typedef struct {
    double slope;
    double intercept;
    double r_squared;
    double std_err;
} LinearRegressionResult;

typedef struct {
    double* coefficients;
    int degree;
    double rss;
    double tss;
} PolynomialRegressionResult;

typedef struct {
    double t_statistic;
    double p_value;
    double df;       // 自由度
    double mean_diff;
} TTestResult;

// 描述性统计
double mean(const double data[], int n);
double mode(double data[], int n);
double variance(const double data[], int n, bool is_sample);
double standard_deviation(const double data[], int n, bool is_sample);
double skewness(const double data[], int n);
double kurtosis(const double data[], int n);

// 回归分析
LinearRegressionResult linear_regression(const double x[], const double y[], int n);
PolynomialRegressionResult polynomial_regression(const double x[], const double y[], int n, int degree);
void free_polynomial_result(PolynomialRegressionResult* result);

// 假设检验
TTestResult t_test(const double sample1[], int n1, const double sample2[], int n2, bool paired);
double chi_square_test(const double observed[], const double expected[], int categories);
double f_test(const double sample1[], int n1, const double sample2[], int n2);

// 相关分析
double pearson_correlation(const double x[], const double y[], int n);
double spearman_rank_correlation(double x[], double y[], int n);

// 概率分布
double normal_pdf(double x, double mu, double sigma);
double normal_cdf(double x, double mu, double sigma);
double t_distribution_pdf(double x, double df);
double t_distribution_cdf(double x, double df);
double chi_squared_pdf(double x, double df);
double chi_squared_cdf(double x, double df);

// 统计工具
void z_normalize(double data[], int n);
double confidence_interval_width(double std_dev, int sample_size, double confidence_level);
double calculate_margin_of_error(double std_dev, int sample_size, double z_score);

// 时间序列分析
double autocorrelation(const double data[], int n, int lag);
double moving_average(const double data[], int n, int window_size);

// 非参数检验
double mann_whitney_u_test(const double sample1[], int n1, const double sample2[], int n2);

void quicksort(double arr[], int left, int right);

// 插值结构体
typedef struct {
    double* x;
    double* y;
    int n;
    double* coefficients;
    int degree;
} Interpolation;

// 微分方程求解器类型
typedef enum {
    EULER,
    RK4,
    ADAMS_BASHFORTH
} ODESolverType;

// 插值方法
typedef double (*MathFunc)(double);
Interpolation* create_interpolation(double x[], double y[], int n, int method);
double interpolate(Interpolation* interp, double xi);
void free_interpolation(Interpolation* interp);

// 数值微分
double numerical_derivative(MathFunc f, double x, double h, int method);
double richardson_extrapolation(MathFunc f, double x, double h, int steps);

// 微分方程求解
//void solve_ode(MathFunc f, double y0, double t_start, double t_end, double step, ODESolverType method, double results[]);

// 偏微分方程求解器（热传导方程示例）
void solve_heat_equation(double (*initial)(double), double L, double T, 
                        double dx, double dt, double alpha, double* solution);

// 数值积分
double adaptive_quadrature(MathFunc f, double a, double b, double tol);

// 最优化方法
double golden_section_search(MathFunc f, double a, double b, double tol);
void conjugate_gradient(double (*f)(double*), double* x0, int n, double tol);

// 非线性方程求解
//double newton_method(MathFunc f, MathFunc df, double x0, double tol);
//double secant_method(MathFunc f, double x0, double x1, double tol);


// 定义复数结构体
typedef struct {
    double real;
    double imag;
} Complex;

// 创建一个复数
Complex create_complex(double real, double imag);

// 复数加法
Complex add_complex(Complex a, Complex b);

// 复数减法
Complex subtract_complex(Complex a, Complex b);

// 复数乘法
Complex multiply_complex(Complex a, Complex b);

// 复数除法
Complex divide_complex(Complex a, Complex b);

// 求复数的模
double modulus_complex(Complex c);

// 打印复数
void print_complex(Complex c);

// 定义多项式结构体
typedef struct {
    int degree;
    double *coefficients;
} Polynomial;

// 创建多项式
Polynomial create_polynomial(int degree, double *coefficients);

// 释放多项式内存
void free_polynomial(Polynomial poly);

// 多项式加法
Polynomial add_polynomials(Polynomial poly1, Polynomial poly2);

// 多项式减法
Polynomial subtract_polynomials(Polynomial poly1, Polynomial poly2);

// 多项式乘法
Polynomial multiply_polynomials(Polynomial poly1, Polynomial poly2);

// 多项式除法（仅返回商）
//Polynomial divide_polynomials(Polynomial poly1, Polynomial poly2);

// 多项式求值
double evaluate_polynomial(Polynomial poly, double x);

// 多项式求导
Polynomial differentiate_polynomial(Polynomial poly);

// 打印多项式
void print_polynomial(Polynomial poly);

// 二项分布概率密度函数
double binomial_pdf(int k, int n, double p);
// 二项分布累积分布函数
double binomial_cdf(int k, int n, double p);

// 泊松分布概率密度函数
double poisson_pdf(int k, double lambda);
// 泊松分布累积分布函数
double poisson_cdf(int k, double lambda);

// 指数分布概率密度函数
double exponential_pdf(double x, double lambda);
// 指数分布累积分布函数
double exponential_cdf(double x, double lambda);

// 冒泡排序
void bubble_sort(int arr[], int n);
// 选择排序
void selection_sort(int arr[], int n);
// 插入排序
void insertion_sort(int arr[], int n);
// 希尔排序
void shell_sort(int arr[], int n);
// 归并排序
void merge_sort(int arr[], int l, int r);
// 快速排序
void quick_sort(int arr[], int low, int high);
// 堆排序
void heap_sort(int arr[], int n);
// 计数排序
void counting_sort(int arr[], int n);
// 桶排序
void bucket_sort(int arr[], int n);
// 基数排序
void radix_sort(int arr[], int n);

// 线性搜索
int linear_search(int arr[], int n, int target);
// 二分搜索
int binary_search(int arr[], int left, int right, int target);
// 插值搜索
int interpolation_search(int arr[], int n, int target);
// 跳跃搜索
int jump_search(int arr[], int n, int target);
// 斐波那契搜索
int fibonacci_search(int arr[], int n, int target);
// 指数搜索
int exponential_search(int arr[], int n, int target);
// 二叉搜索树搜索（假设已有二叉搜索树节点结构体）
typedef struct TreeNode {
    int data;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;
TreeNode* bst_search(TreeNode *root, int target);
// 哈希搜索（简单示例，假设哈希表大小固定）
#define HASH_TABLE_SIZE 100
int hash_search(int hash_table[], int target);
// 顺序搜索（链表实现，假设已有链表节点结构体）
typedef struct ListNode {
    int data;
    struct ListNode *next;
} ListNode;
ListNode* list_search(ListNode *head, int target);
// 分块搜索
int block_search(int arr[], int n, int target, int block_size);

#ifndef FRACTION_H
#define FRACTION_H

// 定义分数结构体
typedef struct {
    int numerator;   // 分子
    int denominator; // 分母
} Fraction;


// 化简分数
Fraction simplify(Fraction f);

// 分数加法
Fraction add(Fraction f1, Fraction f2);

// 分数减法
Fraction subtract(Fraction f1, Fraction f2);

// 分数乘法
Fraction multiply(Fraction f1, Fraction f2);

// 分数除法
Fraction divide(Fraction f1, Fraction f2);

// 打印分数
void printFraction(Fraction f);

#endif // FRACTION_H


#endif // MATH_FUNCTIONS_H