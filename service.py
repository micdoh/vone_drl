class Service:

    def __init__(self, arrival_time, holding_time, nodes=None,
                 nodes_capacity=None, path=None, links_BW=None, links_IS=None):
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.nodes = nodes
        self.nodes_capacity = nodes_capacity
        self.path = path
        self.links_BW = links_BW
        self.links_IS = links_IS

