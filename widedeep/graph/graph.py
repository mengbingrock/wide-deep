

class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []  # 计算图内的节点的列表
        #self.name_scope = None

    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)

    def node_count(self):
        return len(self.nodes)

    def clear_jacobian(self):
        
        """
        清除图中全部节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_jacobian()


default_graph = Graph()



def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

def get_trainable_variables_from_graph(node_name=None, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        import sys
        sys.path.append('..')
        from node.node import Tensor
        return [node for node in graph.nodes if isinstance(node, Tensor) and node.trainable]

    if name_scope:
        node_name = name_scope + '/' + node_name
    return get_node_from_graph(node_name, graph=graph)


def update_node_value_in_graph(node_name, new_value, name_scope=None, graph=None):
    node = get_node_from_graph(node_name, name_scope, graph)
    assert node is not None

    assert node.outputs.shape == new_value.shape
    node.outputs = new_value