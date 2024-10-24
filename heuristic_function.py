from collections import defaultdict
import heapq
import networkx as nx
import matplotlib.pyplot as plt

class CityGraph:
    def __init__(self):
        self.edges = defaultdict(dict)
        self.G = nx.Graph()

    def add_edge(self, from_node, to_node, distance, vehicles):
        self.edges[from_node][to_node] = {
            'distance': distance,
            'cars': vehicles.get('cars', 0),
            'trucks': vehicles.get('trucks', 0),
            'buses': vehicles.get('buses', 0),
            'bikes': vehicles.get('bikes', 0)
        }
        self.edges[to_node][from_node] = self.edges[from_node][to_node]
        self.G.add_edge(from_node, to_node, distance=distance)

    def get_neighbors(self, node):
        return list(self.edges[node].keys())

    def get_edge_info(self, current, next_node):
        return self.edges[current][next_node]

    def get_edge_cost(self, current, next_node):
        return self.edges[current][next_node]['distance']

    def visualize_graph(self, path=None):
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', node_size=1500)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos, width=2)
        edge_labels = nx.get_edge_attributes(self.G, 'distance')
        edge_labels = {k: f'{v} km' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        if path and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color='blue', width=4)

        plt.title("Route Map with Distances")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def a_star_search(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break

        for next_node in graph.get_neighbors(current):
            new_cost = cost_so_far[current] + graph.get_edge_cost(current, next_node)

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + graph.get_edge_cost(current, next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    if goal not in came_from:
        return None, None, None

    path = []
    path_info = []
    current = goal

    while current is not None:
        path.append(current)
        if came_from[current] is not None:
            edge_info = graph.get_edge_info(came_from[current], current)
            path_info.append(edge_info)
        current = came_from.get(current)

    path.reverse()
    path_info.reverse()

    return path, path_info, cost_so_far[goal]

def print_path_details(path, path_info, total_cost):
    if not path:
        print("\nNo path found between these cities.")
        return

    print(f"\nDetailed Path Information:")
    print(f"Complete path: {' -> '.join(path)}")
    print(f"Total distance: {total_cost:.2f} km\n")

    print("Segment details:")
    for i in range(len(path_info)):
        print(f"\nFrom {path[i]} to {path[i + 1]}:")
        info = path_info[i]
        print(f"  Distance: {info['distance']} km")
        print(f"  Cars: {info['cars']}")
        print(f"  Trucks: {info['trucks']}")
        print(f"  Buses: {info['buses']}")
        print(f"  Bikes: {info['bikes']}")

def main():
    city = CityGraph()

    print("\nEnter city connections:")
    num_edges = int(input("Enter number of edges: "))

    for _ in range(num_edges):
        print("\nEdge", _ + 1)
        from_node = input("From city: ")
        to_node = input("To city: ")
        distance = float(input("Distance (km): "))
        cars = int(input("Number of cars: "))
        trucks = int(input("Number of trucks: "))
        buses = int(input("Number of buses: "))
        bikes = int(input("Number of bikes: "))
        city.add_edge(from_node, to_node, distance,
                      {'cars': cars, 'trucks': trucks,
                       'buses': buses, 'bikes': bikes})

    print("\nRoute Planning:")
    start = input("Enter start city: ")
    goal = input("Enter destination city: ")

    path, path_info, total_cost = a_star_search(city, start, goal)

    print_path_details(path, path_info, total_cost)
    city.visualize_graph(path)

if __name__ == "__main__":
    main()
