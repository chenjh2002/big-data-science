# 大数据原理与技术实验三
学号:20337011 姓名:陈俊皓

## 实验要求
+ 实现Google版本的`PageRank`.
+ 从斯坦福大学的SNAP网站（https://snap.stanford.edu/data/index.html）下载一个有向图（directed network）作为数据集，计算得到每一个node的PageRank  score.

## 实验原理
### 原始方法
`PageRank`算法是一种判断有向图中不同节点重要程度的算法。在直观认识中一个节点所具有的价值与其邻接的节点价值密切。相关。因此，`PageRank`算法利用不动点求解的方法，首先建立不同节点之间的概率跳转矩阵`M`，其定义方式如下:
$$
M[i][j]=
\left\{
\begin{aligned}
     &\frac{r_i}{d_i} \qquad (i,j)\in E\\
    &0 \qquad\quad else
\end{aligned}
\right.
$$

其中$E$表示有向图的边集，$r_i$表示节点$i$的`page_score`,$d_i$表示节点$i$的度数。

完成矩阵的构建之后，我们即可建立不动点方程:
$$
    M\cdot \vec{r}=\vec{r}
$$
其中,$\vec{r}$表示有节点的`page_score`组成的列向量。

然而，不动点方程不一定都有解。在当前情景下，只有`M`的每一列列和均为1的情况下，可以得到稳定解。因此，我们需要改进`M`,已让其可以得到稳定解且解的值符合预期。

### `teleport`方法
为了解决上述的问题，我们可以引入随机跳转的理论。具体的实现方法如下，对于每一个节点$i$,其在进行跳转的时候,有$\frac{\beta}{N}$的概率跳到任意一个节点且有$(1-\beta)\cdot\frac{r_i}{d_i}$的概率跳到与其有边连接的节点。

上述方法可以保证`M`矩阵的列和为1，从而保证解的存在性，解决了`Dead-ends`的问题。且，其可以让流量不会困在某一个环中，解决了`Spider-traps`的问题。

### 应对大数据输入的方法
在实际应用`PageRank`算法的过程中，由于节点个数较大，我们的内存不一定能存储整个`M`矩阵。因此，我们需要想办法降低算法的存储损耗。利用网页场景中边较少的特性，我们可以精简`M`矩阵的存储。首先，我们变换求解`page_score`的方程如下:
$$
\begin{aligned}
\vec{r}&=\textbf{A}\vec{r},where\;\textbf{A}_{ji}=\beta \textbf{M}_{ji}+\frac{1-\beta}{N}\\
r_j&=\sum_{i=1}^{N} \textbf{A}_{ji}\cdot r_i\\
r_j&=\sum_{i=1}^{N} [\beta \textbf{M}_{ji}+\frac{1-\beta}{N}]\cdot r_i\\
&=\sum_{i=1}^{N} \beta \textbf{M}_{ji}\cdot r_i+\frac{1-\beta}{N}\sum_{i=1}^{N}r_i\\
&=\sum_{i=1}^{N} \beta \textbf{M}_{ji} r_i+\frac{1-\beta}{N}\\
so\;that\;we\;have:\\
\vec{r}&=\beta\textbf{M}\cdot r+[\frac{1-\beta}{N}]_N
\end{aligned}
$$

由上可得，我们仅需要知道源节点的编号，其对应的度数及其指向的节点即可完成迭代运算。因此，我们可以改进`M`矩阵的存储形式如下:
<table>
    <thead>
        <tr>
            <td>source node</td>
            <td>degreee</td>
            <td>destination nodes
            </td>
        </tr>
    </thead>
</table>

## 实现代码
### 选用的数据集
本次实验选用的数据集为`soc-Epinions1`,共有`75879`个节点以及`508837`属于社交网络型数据库，数据存储形式如下:
```C
0 4
0 5
0 7
0 8
0 9
0 10
0 11
0 12
0 13
...
```
### 算法类
实现`PageRank`算法，首先我们确定运行时所需的参数。包括随机跳转概率的大小，最大迭代轮数，终止循环条件等。一下为定义的算法类:
```c++
const double DEFAULT_ALPHA=0.85;
const double DEFAULT_CONVERGENCE=0.0000001;
const unsigned long DEFAULT_MAX_ITERATION=10000;
const bool DEFAULT_NUMERIC=false;
const string DEFAULT_DELIM=" "; // "=>"

class Table{
private:
    bool trace; //enable tarcing output
    double alpha;//pagerank damping factor
    double convergence;
    unsigned long max_iterations;
    string delim;
    bool numeric;
    vector<size_t> num_outgoing; //number of outgoing links per page
    vector<vector<size_t>> rows;//the rows of the hyperlink matrix
    map<string,size_t>nodes_to_idx;
    map<size_t,string>idx_to_nodes;
    vector<double> pr;//pagerank score

    void trim(string &str);

    template<class Vector,class T> 
    bool insert_into_vector(Vector& v,const T &t);

    void reset();

    size_t insert_mapping(const string& key);

    bool add_arc(size_t from,size_t to);
```

完成参数设置之后，我们需要设计运行`PageRank`算法的主体。具体方法为迭代计算$\vec{r}$,直到其值收敛,实现方法如下:
```C++
void Table::pagerank() {

    vector<size_t>::iterator ci; // current incoming
    double diff = 1;
    size_t i;
    double sum_pr; // sum of current pagerank vector elements
    double dangling_pr; // sum of current pagerank vector elements for dangling
    			// nodes
    unsigned long num_iterations = 0;
    vector<double> old_pr;

    size_t num_rows = rows.size();
    
    if (num_rows == 0) {
        return;
    }
    
    pr.resize(num_rows);

    pr[0] = 1;

    if (trace) {
        print_pagerank();
    }
    
    while (diff > convergence && num_iterations < max_iterations) {

        sum_pr = 0;
        dangling_pr = 0;
        
        for (size_t k = 0; k < pr.size(); k++) {
            double cpr = pr[k];
            sum_pr += cpr;
            if (num_outgoing[k] == 0) {
                dangling_pr += cpr;
            }
        }

        if (num_iterations == 0) {
            old_pr = pr;
        } else {
            /* Normalize so that we start with sum equal to one */
            for (i = 0; i < pr.size(); i++) {
                old_pr[i] = pr[i] / sum_pr;
            }
        }

        /*
         * After normalisation the elements of the pagerank vector sum
         * to one
         */
        sum_pr = 1;
        
        /* An element of the A x I vector; all elements are identical */
        double one_Av = alpha * dangling_pr / num_rows;

        /* An element of the 1 x I vector; all elements are identical */
        double one_Iv = (1 - alpha) * sum_pr / num_rows;

        /* The difference to be checked for convergence */
        diff = 0;
        for (i = 0; i < num_rows; i++) {
            /* The corresponding element of the H multiplication */
            double h = 0.0;
            for (ci = rows[i].begin(); ci != rows[i].end(); ci++) {
                /* The current element of the H vector */
                double h_v = (num_outgoing[*ci])
                    ? 1.0 / num_outgoing[*ci]
                    : 0.0;
                if (num_iterations == 0 && trace) {
                    cout << "h[" << i << "," << *ci << "]=" << h_v << endl;
                }
                h += h_v * old_pr[*ci];
            }
            h *= alpha;
            pr[i] = h + one_Av + one_Iv;
            diff += fabs(pr[i] - old_pr[i]);
        }
        num_iterations++;
        if (trace) {
            cout << num_iterations << ": ";
            print_pagerank();
        }
    }
    
}
```

## 实验结果
经过`PageRank`算法后得到的`page_score`如下:
```C
0 = 6.11292326762618e-06
4
 = 3.51254818835252e-05
5
 = 4.17938306690565e-05
7
 = 1.71778541103321e-05
8
 = 4.91026098571742e-05
9
 = 7.83911605839447e-06
10
 = 7.94015486122371e-05
11
 = 2.99254597880927e-05
12
 = 0.000136514207701607
13
 = 0.000151929835611059
14
 = 8.86454021105371e-05
15
 = 2.01565842803718e-05
16
 = 1.68778504300123e-05
17
 = 0.000121850027685361
18
 = 0.0019181471571822
19
 = 7.75210940428979e-05
20
 = 7.37260092882207e-05
21
 = 0.000109282035316866
22
 = 7.92132454358385e-05
23
 = 1.33894793345448e-05
24
 = 5.53638087386916e-05
25
 = 4.04434018555537e-05
26
 = 6.42077277090811e-05
27
 = 0.000382704332367135
28
 = 0.000206958828466285
29
 = 9.21671928409508e-05
30
 = 9.54332505320112e-05
31
 = 0.000165862680791486
32
 = 8.526279293503e-05
33
 = 5.60038318947634e-05
34
 = 0.000178721421401169
35
 = 0.000129430893214344
36
 = 4.93486729058325e-05
37
 = 3.66947769302052e-05
38
 = 2.71362792801046e-05
39
 = 8.06473369023083e-05
40
 = 0.000217379936958211
 ...
```

## 实验心得
通过此次实验，我了解到`PageRank`算法的原理及其实现过程。