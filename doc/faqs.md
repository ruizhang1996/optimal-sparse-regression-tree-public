# Frequently Asked Questions
- [Does OSRT (implicitly) restrict the depth of the resulting tree?](##depth)
- [If the regularization parameter is set to zero, does the algorithm search for a decision tree of arbitrary size that has perfect accuracy? ](#perfect_accuracy)
- [Why does OSRT run for a long time when the regularization parameter (lambda) is set to zero?](##long_run)
- [Is there a way to limit the size of produced tree?](##limit_tree_size)
- [In general, how does OSRT set the regularization parameter?](##set_lambda)

---

## Does OSRT (implicitly) restrict the depth of the resulting tree? 

No, OSRT does not restrict the depth of the resulting tree unless you set the depth limit in configuration. Our sparsity constraint is from the regularization parameter (lambda) which is used to penalize the number of leaves. If lambda is set to a large value, e.g. lambda=1, then the generated tree will be a root node without any split and the prediction at this point will be the mean of samples (Suppose the metrics is L2).  When lambda becomes smaller, the generated trees will have more leaves. But the number of leaves doesn't guarantee what depth a tree has since OSRT generates trees of any shape.



## If the regularization parameter is set to zero, does the algorithm search for a decision tree of arbitrary size that has perfect accuracy? 

If the regularization parameter (lambda) is set to 0, OSRT tends to find a decision tree of arbitrary size that has the best possible accuracy. If the dataset doesn't have equivalent samples (that is samples with the same feature values but different targets), then the best possible accuracy is 1 (loss=0). Otherwise, the best possible loss will be non-zero.



## Why does OSRT run for a long time when the regularization parameter (lambda) is set to zero?

The running time depends on the dataset itself and the regularization parameter (lambda). In general, setting lambda to 0 will make the running time longer. Setting lambda to 0 is kind of deactivating the branch-and-bound in OSRT. In other words, we are kind of using brute force to search over the whole space without effective pruning, though dynamic programming can help for computational reuse. 
In OSRT, we compare the difference between the upper and lower bound of a subproblem with lambda to determine whether this subproblem needs to be further split. If lambda=0, we can always split a subproblem. Therefore, it will take more time to run.  Actually, it doesn't make sense to set lambda smaller than 1/n, where n is the number of samples.



## Is there a way to limit the size of produced tree?

Regularization parameter (lambda) is used to limit the size of the produced tree (specifically, in OSRT, it limits the number of leaves of the produced tree). We usually set lambda to [0.1, 0.05, 0.01, 0.005, 0.001], but the value really depends on the dataset. One thing that might be helpful is considering how many samples should be captured by each leaf node. Suppose you want each leaf node to contain at least 10 samples. Then setting the regularization parameter to 10/n is reasonable. In general, the larger the value of lambda is, the sparser a tree you will get.



## In general, how does OSRT set the regularization parameter? 

OSRT aims to find an optimal tree that minimizes the training loss with a penalty on the number of leaves. The mathematical description is min loss+lambda*# of leaves. When we run OSRT, we usually set lambda to different non-zero values and usually not smaller than 1/n. 
