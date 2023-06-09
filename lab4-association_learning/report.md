# 大数据原理与技术第五次实验
学号:20337011 姓名:陈俊皓

## 实验要求
+ 找到对应的含有实验的代码+数据集的关联规则论文（Google关键词查找Associate rule，不考虑年份），阅读论文。
+ 复现代码并与论文实验结果进行对比。
+ 写一页的阅读+代码报告，描述你的复现结果以及发现。

## 实验过程

### 论文标题`AN IMPROVED APRIORI ALGORITHM FOR ASSOCIATION RULES`

#### 针对问题
在传统的`apriori`算法中，有大量的时间被用于挑选以及存储候选集中。如果可以降低挑选候选集的时间复杂度以及降低其存储的空间消耗，将大幅提升原算法的性能。

#### 解决思路
`apriori`算法的理论根据是如果一个`itemset`的子集不是`frequency set`,那么他必定不是`frequency set`。因此在迭代地构建`candidate set`过程中，可以仅保留比起元素个数少的集合中为`frequency set`的子集以降低时间复杂度以及空间消耗。但是，对于`itemset`较大以及`transaction`较多的情形，由于`apriori`算法需要多次重复查找数据库，其性能仍然较低。

由于一个`itemset`往往不会频繁的出现在`transaction`中，如果我们可以降低每次构建候选集所需查找的`transaction`个数，将提升原算法的时间消耗。根据`apriori`的理论基础，我们可以利用`self-join`的方式构建候选集。具体的实现过程如下:

1. 扫描所有的`transaction`生成$L_1$,并存储其对应的物件，支持数及其对应出现在哪些`transaction`中(items,support,transaction IDs)。

2. 利用`self-join`的方式构建$C_k$。

3. 利用$L_1$定位需要查找的事务(key)。

4. 扫描目标事务(`target transaction`)生成候选集$L_k$。

此处叙述第`3`步的实现思路。在构建了$C_k$之后，我们遍历其中的元素，通过$L_1$可以得到其中`support`最小的元素，查找该元素出现对应的`transaction`即可判断该候选集是否为频繁集。


### 复现过程

#### 选择的数据集
此次复现实验选取了`Kaggle`上的`Market_Basket_Optimisation.csv`数据，其数据形式如下:
```C
shrimp,almonds,avocado,vegetables mix,green grapes,whole weat flour,yams,cottage cheese,energy drink,tomato juice,low fat yogurt,green tea,honey,salad,mineral water,salmon,antioxydant juice,frozen smoothie,spinach,olive oil
burgers,meatballs,eggs
chutney
turkey,avocado
mineral water,milk,energy bar,whole wheat rice,green tea
low fat yogurt
whole wheat pasta,french fries
soup,light cream,shallot
frozen vegetables,spaghetti,green tea
french fries
eggs,pet food
cookies
turkey,burgers,mineral water,eggs,cooking oil
spaghetti,champagne,cookies
mineral water,salmon
mineral water
shrimp,chocolate,chicken,honey,oil,cooking oil,low fat yogurt
turkey,eggs
...
```

#### 定义事务管理器

针对输入的事务集合，我们需要存储构建$L_1$过程中所需要的数据，包括存储了所有物品种类的集合`items`,针对每一个物品其出现在的事务`transaction_index_map`以及事务总数。该类还需提供计算`support`以及加入新事务的功能。

```Python
class TransactionManager(object):
    def __init__(self,transactions):
        self.__num_transaction=0
        self.__items=[]
        self.__transaction_index_map={}
        
        for transaction in transactions:
            self.add_transaction(transaction)
    
    def add_transaction(self,transaction):
        for item in transaction:
            if item not in self.__transaction_index_map:
                self.__items.append(item)
                self.__transaction_index_map[item]=set()
            
            self.__transaction_index_map[item].add(self.__num_transaction)
        
        self.__num_transaction+=1
        
    def calc_support(self,items):
        # Empty items is supported by all transaction
        if not items:
            return 1.0
        
        # Empty transaction support no items
        if not self.__num_transaction:
            return 0.0
        
        # Create the transaction index itersection
        sum_indexes=None
        for item in items:
            indexes=self.__transaction_index_map.get(item)
            
            if indexes is None:
                return 0.0
            
            if sum_indexes is None:
                sum_indexes=indexes
            else:
                sum_indexes=sum_indexes.intersection(indexes)


        
        return float(len(sum_indexes))/self.__num_transaction
    
    
                
    def initial_candidates(self):
        return [frozenset([item]) for item in self.__items]
    
    @property
    def num_transaction(self):
        return self.__num_transaction
    
    @property
    def items(self):
        return sorted(self.__items)
    
    @staticmethod
    def create(transactions):
        if isinstance(transactions,TransactionManager):
            return transactions
        return TransactionManager(transactions)
```

#### 设置存储格式
在实验中需要存储候选集，频繁集以及关联规则的记录。
```Python
SupportRecord=namedtuple(
    'SupportRecord',('items','support'))
RelationRecord=namedtuple(
    'RelationRecord',SupportRecord._fields+('ordered_statistics',))
OrderedStatistic=namedtuple(
    'OrderedStatistic',('item_base','item_add','confidence','lift',))
```

#### 通过$L_{k-1}$创建$C_k$
```Python
def create_next_candidates(prev_candidates,length):
    """
    Return the aprioi candidates as a list
    
    Arguments:
        prev_candidates -- Previous candidates as a list
        length -- The lengths of the next candidates
    """
    
    # Solve the items
    items=sorted(frozenset(chain.from_iterable(prev_candidates)))
    # print(items)
    
    # Create the temporary candidates. These will be filtered below
    tmp_next_candidates=(frozenset(x) for x in combinations(items,length))
    
    # Return all the candidates if the length of the next candidates is 2
    # because their subsets are the same as items.
    if length<3:
        return list(tmp_next_candidates)
    
    # Filter candidates that all of their subsets are
    # in the previous candidates
    next_candidates=[
        candidate for candidate in tmp_next_candidates
        if all(
            frozenset(x) in prev_candidates
            for x in combinations(candidate,length-1))
    ]
    
    # print(next_candidates)
    
    return next_candidates
```

#### 递归的获取所有频繁集
```Python
def gen_support_records(transaction_manager:TransactionManager,min_support,**kwargs):
    """
    Returns a generator of support records with given transactions.
    
    Arguments:
        transaction_manager  --Transactions as a TransactionManager instance.
        min_support -- A minimum support(float).
    
    Keyword arguments:
        max_length --The maximum length of realtions (integer).
    
    """
    
    # Parse arguments.
    max_length=kwargs.get('max_length')
    
    # For testing
    _create_next_candidates=kwargs.get(
        '_create_next_candidates',create_next_candidates)
    
    # Process.
    candidates=transaction_manager.initial_candidates()
    
    length=1
    while candidates:
        relations=set()
        for relation_candidate in candidates:
            support=transaction_manager.calc_support(relation_candidate)
            
            # print(support)
            if support <min_support:
                continue
            
            candidate_set=frozenset(relation_candidate)
            relations.add(candidate_set)
            yield SupportRecord(candidate_set,support)
        
        length+=1
        # print(relations)
        
        if max_length and length>max_length:
            break
        # print(f"length:{length}")
        candidates=_create_next_candidates(relations,length)

```


#### 生成关联规则
```Python
def gen_ordered_statistics(transcation_manager:TransactionManager,record):
    """
    Returns a generator of ordered statistics as OrderedStatistic instances.
    
    Arguments:
        transaction_manager -- Transactions as a TransactionManager instance.
        record -- A support as a SupportRecord instance.
    
    """
    
    items=record.items
    sorted_items=sorted(items)
    
    for base_length in range(len(items)):
        for combination_set in combinations(sorted_items,base_length):
            items_base=frozenset(combination_set)
            items_add=frozenset(items.difference(items_base))
            confidence=(
                record.support/transcation_manager.calc_support(items_base))
            
            lift=confidence/ transcation_manager.calc_support(items_add)
            # print(frozenset(items_base))
            
            yield OrderedStatistic(
                frozenset(items_base),frozenset(items_add),confidence,lift
            )

def filter_ordered_statistics(ordered_statistics,**kwargs):
    """
    Filter OrderedStatistic objects
    
    Arguments:
        ordered_statistics -- A OrderedStatistic iterable object.
    
    Keyword arguments
        min_confidence -- The minimum confidence of the relations(float).
        min_lift -- The minimum lift of relations (float).
     
    """
    
    min_confidence=kwargs.get('min_confidence',0.0)
    min_lift=kwargs.get('min_lift',0.0)
    
    for ordered_statistic in ordered_statistics:
        # print(f"ordered statistic lift: {ordered_statistic.lift}")
        if ordered_statistic.confidence <min_confidence:
            continue
        
        if ordered_statistic.lift<min_lift:
            continue
        
        yield ordered_statistic
```

#### `m_apriori`函数
```Python
def apriori(transactions,**kwargs):
    """
    Executes Apriori algorithm and returns a RelationRecord generator.

    Arguments:
        transactions -- A transaction iterable object
                        (eg. [['A', 'B'], ['B', 'C']]).

    Keyword arguments:
        min_support -- The minimum support of relations (float).
        min_confidence -- The minimum confidence of relations (float).
        min_lift -- The minimum lift of relations (float).
        max_length -- The maximum length of the relation (integer).
    
    """
    
    # Parse the arguments.
    min_support = kwargs.get('min_support', 0.1)
    min_confidence = kwargs.get('min_confidence', 0.0)
    min_lift = kwargs.get('min_lift', 0.0)
    max_length = kwargs.get('max_length', None)   
    
    # Check arguments.
    if min_support <= 0:
        raise ValueError('minimum support must be > 0')

    # For testing.
    _gen_support_records = kwargs.get(
        '_gen_support_records', gen_support_records)
    _gen_ordered_statistics = kwargs.get(
        '_gen_ordered_statistics', gen_ordered_statistics)
    _filter_ordered_statistics = kwargs.get(
        '_filter_ordered_statistics', filter_ordered_statistics)
    
     # Calculate supports.
    print("Creating Transaction Set")
    transaction_manager = TransactionManager.create(transactions)
    support_records = _gen_support_records(
        transaction_manager, min_support, max_length=max_length)

    # Calculate ordered stats.
    print("Creating Support Records")
    for support_record in support_records:
        ordered_statistics = list(
            _filter_ordered_statistics(
                _gen_ordered_statistics(transaction_manager, support_record),
                min_confidence=min_confidence,
                min_lift=min_lift,
            )
        )
        # print(ordered_statistics)
        if not ordered_statistics:
            continue
        yield RelationRecord(
            support_record.items, support_record.support, ordered_statistics)
```


## 实验结果
在实现了上述算法后，我们可以得到关联规则如下:
```json
{"items": ["shrimp"], "support": 0.07145713904812692, "ordered_statistics": [{"item_base": [], "item_add": ["shrimp"], "confidence": 0.07145713904812692, "lift": 1.0}]}
{"items": ["almonds"], "support": 0.020397280362618318, "ordered_statistics": [{"item_base": [], "item_add": ["almonds"], "confidence": 0.020397280362618318, "lift": 1.0}]}
{"items": ["avocado"], "support": 0.03332888948140248, "ordered_statistics": [{"item_base": [], "item_add": ["avocado"], "confidence": 0.03332888948140248, "lift": 1.0}]}
{"items": ["vegetables mix"], "support": 0.025729902679642713, "ordered_statistics": [{"item_base": [], "item_add": ["vegetables mix"], "confidence": 0.025729902679642713, "lift": 1.0}]}
{"items": ["yams"], "support": 0.011465137981602452, "ordered_statistics": [{"item_base": [], "item_add": ["yams"], "confidence": 0.011465137981602452, "lift": 1.0}]}
{"items": ["cottage cheese"], "support": 0.03186241834422077, "ordered_statistics": [{"item_base": [], "item_add": ["cottage cheese"], "confidence": 0.03186241834422077, "lift": 1.0}]}
{"items": ["energy drink"], "support": 0.026663111585121985, "ordered_statistics": [{"item_base": [], "item_add": ["energy drink"], "confidence": 0.026663111585121985, "lift": 1.0}]}
{"items": ["tomato juice"], "support": 0.030395947207039063, "ordered_statistics": [{"item_base": [], "item_add": ["tomato juice"], "confidence": 0.030395947207039063, "lift": 1.0}]}
{"items": ["low fat yogurt"], "support": 0.07652313024930009, "ordered_statistics": [{"item_base": [], "item_add": ["low fat yogurt"], "confidence": 0.07652313024930009, "lift": 1.0}]}
{"items": ["green tea"], "support": 0.13211571790427942, "ordered_statistics": [{"item_base": [], "item_add": ["green tea"], "confidence": 0.13211571790427942, "lift": 1.0}]}
{"items": ["honey"], "support": 0.047460338621517134, "ordered_statistics": [{"item_base": [], "item_add": ["honey"], "confidence": 0.047460338621517134, "lift": 1.0}]}
{"items": ["mineral water"], "support": 0.23836821757099053, "ordered_statistics": [{"item_base": [], "item_add": ["mineral water"], "confidence": 0.23836821757099053, "lift": 1.0}]}
{"items": ["salmon"], "support": 0.04252766297826956, "ordered_statistics": [{"item_base": [], "item_add": ["salmon"], "confidence": 0.04252766297826956, "lift": 1.0}]}
{"items": ["frozen smoothie"], "support": 0.06332489001466471, "ordered_statistics": [{"item_base": [], "item_add": ["frozen smoothie"], "confidence": 0.06332489001466471, "lift": 1.0}]}
{"items": ["olive oil"], "support": 0.0658578856152513, "ordered_statistics": [{"item_base": [], "item_add": ["olive oil"], "confidence": 0.0658578856152513, "lift": 1.0}]}
{"items": ["burgers"], "support": 0.0871883748833489, "ordered_statistics": [{"item_base": [], "item_add": ["burgers"], "confidence": 0.0871883748833489, "lift": 1.0}]}
{"items": ["meatballs"], "support": 0.020930542594320756, "ordered_statistics": [{"item_base": [], "item_add": ["meatballs"], "confidence": 0.020930542594320756, "lift": 1.0}]}
{"items": ["eggs"], "support": 0.17970937208372217, "ordered_statistics": [{"item_base": [], "item_add": ["eggs"], "confidence": 0.17970937208372217, "lift": 1.0}]}
{"items": ["turkey"], "support": 0.06252499666711105, "ordered_statistics": [{"item_base": [], "item_add": ["turkey"], "confidence": 0.06252499666711105, "lift": 1.0}]}
{"items": ["milk"], "support": 0.12958272230369283, "ordered_statistics": [{"item_base": [], "item_add": ["milk"], "confidence": 0.12958272230369283, "lift": 1.0}]}
...
```

## 实验心得
通过此次实验，我对于关联规则地各过程实现方法更加清晰，并对其一些优化的方式有了自己的认识。

