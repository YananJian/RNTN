import numpy as np
import parsePTB
class RNTN:

    def __init__(self, wordvec_dim, num_words, output_dim, words, minibatch_size = 10, learning_rate = 0.0001):
        self.wordvec_dim = wordvec_dim
        self.num_words = num_words
        self.words = words
        self.output_dim = output_dim
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self._lambda = 0.0001

    def init(self):
        # word embedding
        # each word dim: d
        self.X = 0.01*np.random.randn(self.wordvec_dim, self.num_words)
        # hidden weights
        # hidden input dim: d*2d
        # hidden output to next hidden, dim: (d*2d)*2d
        self.W = 0.01*np.random.randn(self.wordvec_dim, self.wordvec_dim*2)
        self.b = np.zeros(self.wordvec_dim)
        # softmax weights
        self.Sw = 0.01*np.random.randn(self.output_dim, self.wordvec_dim)
        self.Sb = np.zeros(self.output_dim)

        self.dSw = np.zeros(self.Sw.shape)
        self.dSb = np.zeros(self.Sb.shape)
        self.db = np.zeros(self.b.shape)
        self.dW = np.zeros(self.W.shape)
        self.dX = {}
        for i in range(0, self.num_words):
            self.dX[i] = np.zeros(self.wordvec_dim)
    
    def forwardPropagateTree(self, node):
        cost = 0
        correct = 0
        total = 0
        if node.isLeaf:
            node.node_vec = self.X[:, self.words.index(node.word)]
        else:
            cst, crt, ttl = self.forwardPropagateTree(node.left)
            cost += cst
            correct += crt
            total += ttl
                
            cst, crt, ttl = self.forwardPropagateTree(node.right)
            cost += cst
            correct += crt
            total += ttl

            stacked_vec = np.hstack([node.left.node_vec, node.right.node_vec])
            node.node_vec = np.dot(self.W, stacked_vec) + self.b
            node.node_vec = np.tanh(node.node_vec)

        node.out_prob = self.softmax(node)
        if np.argmax(node.out_prob) == node.label:
            correct += 1
        total += 1
        return cost - np.log(node.out_prob[node.label]), correct, total

    def softmax(self, node):
        x = np.dot(self.Sw, node.node_vec) + self.Sb
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def backPropagateTree(self, node, err=None):
        #calculate prediction diff
        prediction_diff = node.out_prob
        prediction_diff[node.label] -= 1

        self.dSw += np.outer(prediction_diff, node.node_vec)
        self.dSb += prediction_diff

        prediction_diff = self.Sw.T.dot(prediction_diff)
        self.dZ_pred = prediction_diff*(1 - node.node_vec**2)

        self.dZ_full = self.dZ_pred

        if err is not None and err != []:
            self.dZ_full = self.dZ_pred + err

        if node.isLeaf:
            self.dX[self.words.index(node.word)] += self.dZ_full
            return

        # Hidden grad
        else:
            parent_vec = np.hstack([node.left.node_vec, node.right.node_vec])
            self.dW += np.outer(self.dZ_full, parent_vec)
            self.db += self.dZ_full
            
            # Error signal to children
            self.dZ_full = np.dot(self.W.T, self.dZ_full) 
            self.dZ_full = self.dZ_full * (1 - parent_vec**2)
            self.backPropagateTree(node.left, self.dZ_full[:self.wordvec_dim])
            self.backPropagateTree(node.right,self.dZ_full[self.wordvec_dim:])

    def clearGridents(self):
        self.dW.fill(0)
        self.db.fill(0)
        self.dSw.fill(0)
        self.dSb.fill(0)
        for i in range(0, self.num_words):
            self.dX[i] = np.zeros(self.wordvec_dim)

    def step(self, minibatch):
        cost = 0
        correct = 0
        total = 0
        self.clearGridents()
        for tree in minibatch:
            cst, crr, ttl = self.forwardPropagateTree(tree)
            cost += cst
            correct += crr
            total += ttl

        for tree in minibatch:
            self.backPropagateTree(tree)
        
        scale = (1./self.minibatch_size)
        for v in self.dX.itervalues():
            v *=scale
        
        for j in self.dX.iterkeys():
            self.X[:,j] += scale*self.dX[j] 
        cost = self.getMinibatchCost()
        self.updateMinibatchGradient()
        if total != 0:
            print 'cost:', scale*cost, ", acc: %0.2f"%((correct*1.0/total)*100)+"%" 

    def getMinibatchCost(self):
        cost = self._lambda/2 * np.linalg.norm(self.W)**2
        cost += self._lambda/2 * np.linalg.norm(self.X)**2
        cost += self._lambda/2 * np.linalg.norm(self.Sw)**2
        return cost

    def updateMinibatchGradient(self):
        scale = 1.0/self.minibatch_size
        # average gradients and regularize them
        self.W = self.W - self.learning_rate*( scale*self.dW + self._lambda * self.W )
        self.Sw = self.Sw - self.learning_rate*(scale*self.dSw + self._lambda * self.Sw)
        self.b = self.b - self.learning_rate*(scale*self.db)
        self.Sb = self.Sb - self.learning_rate*(scale*self.dSb)
        
if __name__ == "__main__":
    file = 'trees/train.txt'
    p = parsePTB.Parser()
    trees = []
    words = set()
    with open(file,'r') as fid:
        lines = fid.readlines()
        for l in lines:
            node = p.parse(l)
            trees.append(node)
            parsePTB.findWords(node, words)
    words = list(words)
    model = RNTN(30, len(words), 5, words, 100, 0.00001)
    model.init()
    minibatch_size = 30
    epoch = 10
    for e in xrange(0, epoch, 1):
        print "****** epoch : ", e
        for i in xrange(0, len(words)-minibatch_size+1, minibatch_size):
            print "------------ iter :", i
            model.step(trees[i:i+minibatch_size])


            



        
