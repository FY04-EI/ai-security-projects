import math

class Value:

    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.child = set()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        out.child = (self, other)

        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward

        return(out)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        out.child = (self, other)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return(out)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self*(other**(-1))

    def tanh(self):
        out = Value(math.tanh(self.data))
        out.child = (self, )

        def _backward():
            self.grad += (1 - (out.data)**2) * out.grad
        out._backward = _backward

        return(out)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other)
        out.child = (self,)

        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad
        out._backward = _backward

        return(out)

    def exp(self):
        out = Value(math.exp(self.data))
        out.child = (self, )

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return(out)

    def backward(self):
        self.grad = 1.0
        visited = set()
        out = []
        def topo_order(node):
            if node not in visited :
                visited.add(node)
                for child in node.child :
                    topo_order(child)
                out.append(node)
            return()
        topo_order(self)
        for node in reversed(out):
            node._backward()
        return
