import numpy as np
import random
# group the list into pairs (randomly make 1 triplet)
def grp(lst):
    n = len(lst)

    # make pairs if even number of elements
    if n % 2 == 0:
        for i in range(0, n, 2):
            val = lst[i:i+2]
            yield list(val)

    # randomly make a triplet if odd
    else:
        r = random.choice(lst)
        i = 0
        while i < n:
            if r in lst[i:i+3]:
                val = lst[i:i+3]
                i += 3
            else:
                val = lst[i:i+2]
                i += 2
            yield list(val)

def group(lst):
    return list(grp(lst))

# interpolates between x1 and x2 as a predictor of y
def interp(x1,x2,y):
    num = (y-x2)*(x1-x2)
    denom = (x1-x2)**2
    numtot = np.sum(num)
    denomtot = np.sum(denom)

    a = (numtot-num)/(denomtot-denom)
    a = np.where(a < 1, a, 1)
    a = np.where(a > 0, a, 0)
    return a

# interpolates between x1, x2, and x3
# drops the variable with the least predictive power
def interp3(x1,x2,x3,y):
    a12 = interp(x1,x2,y)
    a23 = interp(x2,x3,y)
    a13 = interp(x1,x3,y)
    vals = [np.mean(a12+a13),np.mean(1-a12+a23),np.mean(1-a23+1-a13)]
    drop = np.argmin(vals)

    if drop == 0:
        a = np.zeros(len(y))
        b = a23
    elif drop == 1:
        a = a13
        b = np.zeros(len(y))
    else:
        a = a12
        b = 1-a

    return (a,b)

# preprocess a new observation
def prep_new(xtrain,xtest,ytrain):
    q = np.shape(xtrain)[1]

    sum0 = np.mean(xtrain[ytrain == -1],0)
    sum1 = np.mean(xtrain[ytrain == 1],0)
    l = 0.5*(sum0+sum1)

    xadjs = xtrain-l
    s = np.array([1/np.mean(xadjs[:,i][ytrain == 1]) for i in range(q)])

    out = s*(xtest-l) # np.dot
    return(out,s,l)

# preprocess a data matrix
def prep_mat(x,y):
    q = np.shape(x)[1]
    n = np.shape(x)[0]

    n0 = len(y[y == -1])
    n1 = len(y[y == 1])
    n = n0 + n1
    sum0 = np.sum(x[y == -1],0)
    sum1 = np.sum(x[y == 1],0)

    x0 = np.array([np.zeros(q) if y[k] == 1 else x[k] for k in range(n)])
    x1 = np.array([np.zeros(q) if y[k] == -1 else x[k] for k in range(n)])
    n0adj = n0 - np.array([0 if k == 0 else 1 for k in x0[:,0]])
    n1adj = n1 - np.array([0 if k == 0 else 1 for k in x1[:,0]])
    n0adj.shape = (n,1)
    n1adj.shape = (n,1)
    adj0 = (sum0-x0)/n0adj
    adj1 = (sum1-x1)/n1adj
    l = (adj0 + adj1)/2

    xadjs = [x-k for k in l]
    s = [1/np.mean(k[y != y[i]],0) for i,k in enumerate(xadjs)]
    s = np.array([-k if y[i] == 1 else k for i,k in enumerate(s)])
    return(s*(x-l))

# tournament classifier iteration
def tc_it(x,y):
    q = np.shape(x)[1]
    n = np.shape(x)[0]
    adj = np.copy(x)
    coefs = np.zeros(shape = (n,q))+1
    pairs = group(range(q))
    itr = 1

    while len(pairs[0])> 1:
        i = 0
        j = 0
        for k, pair in enumerate(pairs):
            if len(pair) == 3:
                a,b = interp3(adj[:,i],adj[:,i+1],adj[:,i+2],y)

                if itr > 1:
                    a.shape = (n,1)
                    b.shape = (n,1)
                coefs[:,pair[0]] = coefs[:,pair[0]]*a
                coefs[:,pair[1]] = coefs[:,pair[1]]*b
                coefs[:,pair[2]] = coefs[:,pair[2]]*(1-a-b)

                if itr > 1:
                    a.shape = (n,)
                    b.shape = (n,)
                adj[:,j] = adj[:,i]*a + adj[:,i+1]*b+adj[:,i+2]*(1-a-b)
                i += 3

                if itr > 1:
                    pairs[k] = pair[0] + pair[1] + pair[2]
            elif len(pair) == 2:
                a = interp(adj[:,i],adj[:,i+1],y)

                if itr > 1:
                    a.shape = (n,1)
                coefs[:,pair[0]] = coefs[:,pair[0]]*a
                coefs[:,pair[1]] = coefs[:,pair[1]]*(1-a)

                if itr > 1:
                    a.shape = (n,)
                adj[:,j] = adj[:,i]*a + adj[:,i+1]*(1-a)

                i += 2
                if itr > 1:
                    pairs[k] = pair[0] + pair[1]
            j += 1

        itr += 1
        pairs = group(pairs)

    return (np.mean(coefs,0)) #,adj[:,0])

# inverse permutation function
def invert_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

# tournament classifier
def tc_coefs(x,y,iters):
    q = np.shape(x)[1]
    n = np.shape(x)[0]

    for i in range(iters):
        perm = np.random.permutation(range(q))
        coefs_perm = tc_it(x[:,perm],y) #[0]
        try:
            coefs += coefs_perm[invert_permutation(perm)]
        except NameError:
            coefs = coefs_perm[invert_permutation(perm)]

    coefs /= iters
    #preds = [1 if k > 0 else -1 for k in np.sum(coefs*x,1)]
    #acc = np.sum(y == preds)/n
    return coefs

def tc_pred(x,y,coefs):
    n = np.shape(x)[0]
    preds = [1 if k > 0 else -1 for k in np.sum(coefs*x,1)]
    acc = np.sum(y == preds)/n
    return(preds,acc)

def tc(x,y,iters):
    coefs = tc_coefs(x,y,iters)
    preds,acc = tc_pred(x,y,coefs)
    return(coefs,preds,acc)
