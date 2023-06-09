#include<vector>
#include<set>
#include<map>
#include<string>
#include<list>

using namespace std;

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

public:
    Table(double a=DEFAULT_ALPHA,double c=DEFAULT_CONVERGENCE,
            size_t i=DEFAULT_MAX_ITERATION,bool t=false,
            bool n=DEFAULT_NUMERIC,
            string d=DEFAULT_DELIM);
    
    void reserve(size_t size);

    const size_t get_num_rows();

    void set_num_rows(size_t num_rows);

    const void error(const char*p,const char *p2="");

    int read_file(const string &filename);

    void pagerank();

    const vector<double>& get_pagerank();

    const string get_node_name(size_t index);

    const map<size_t,string> &get_mapping();

    const double get_alpha();

    void set_alpha(double a);

    const unsigned long get_max_iterations();

    void set_max_iterations(unsigned long i);

    const double get_convergence();

    void set_convergence(double c);

    const bool get_trace();

    void set_trace(bool t);

    const bool get_numeric();

    void set_numeric(bool n);

    const string get_delim();

    void set_delim(string d);

    const void print_params(ostream &out);

    const void print_table();

    const void print_outgoing();

    const void print_pagerank();

    const void print_pagerank_v();
};
