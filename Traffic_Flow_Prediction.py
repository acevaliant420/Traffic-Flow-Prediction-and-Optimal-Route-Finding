import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class TrafficPredictor:
    def __init__(self):
        self.rfc = RandomForestClassifier(random_state=0)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, data_path):
        data = pd.read_csv(data_path)
        data['Traffic Situation'] = data['Traffic Situation'].replace(
            {'low': 1, 'normal': 2, 'high': 3, 'heavy': 4}
        )
        data['Day of the week'] = data['Day of the week'].replace(
            {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
             'Friday': 5, 'Saturday': 6, 'Sunday': 7}
        )

        data['hour'] = pd.to_datetime(data['Time']).dt.hour
        data['minute'] = pd.to_datetime(data['Time']).dt.minute
        data['AM/PM'] = (data['Time'].str.split().str[1] == 'PM').astype(int)
        data = data.drop(columns=['Time'], axis=1)

        X = data[['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount',
                  'TruckCount', 'Total', 'hour', 'minute', 'AM/PM']]
        y = data['Traffic Situation'].values

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.rfc.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, day, time, vehicles):
        if not self.is_trained:
            raise Exception("Model needs to be trained first")

        hour, minute, second = map(int, time.split(':'))
        am_pm = 1 if hour >= 12 else 0
        hour = hour % 12

        total_vehicles = sum(vehicles.values())
        input_data = pd.DataFrame([[
            0,
            day,
            vehicles.get('cars', 0),
            vehicles.get('bikes', 0),
            vehicles.get('buses', 0),
            vehicles.get('trucks', 0),
            total_vehicles,
            hour,
            minute,
            am_pm
        ]], columns=['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount',
                     'TruckCount', 'Total', 'hour', 'minute', 'AM/PM'])

        input_scaled = self.scaler.transform(input_data)
        return self.rfc.predict(input_scaled)[0]


class TrafficAwareCityGraph:
    def __init__(self):
        self.edges = defaultdict(dict)
        self.G = nx.Graph()
        self.traffic_predictor = TrafficPredictor()

    def train_traffic_model(self, data_path):
        self.traffic_predictor.train(data_path)

    def add_edge(self, from_node, to_node, distance, vehicles):
        self.edges[from_node][to_node] = {
            'distance': distance,
            'vehicles': vehicles
        }
        self.edges[to_node][from_node] = self.edges[from_node][to_node]
        self.G.add_edge(from_node, to_node, distance=distance)

    def get_traffic_weight(self, traffic_level):
        weights = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}
        return weights.get(traffic_level, 1.0)

    def get_edge_cost(self, current, next_node, day, time):
        edge_info = self.edges[current][next_node]
        base_distance = edge_info['distance']

        traffic_level = self.traffic_predictor.predict(
            day,
            time,
            edge_info['vehicles']
        )

        traffic_weight = self.get_traffic_weight(traffic_level)
        return base_distance * traffic_weight

    def get_neighbors(self, node):
        return list(self.edges[node].keys())

    def a_star_search(self, start, goal, day, time):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current == goal:
                break

            for next_node in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.get_edge_cost(
                    current, next_node, day, time
                )

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.get_edge_cost(current, next_node, day, time)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        if goal not in came_from:
            return None, None

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()

        return path, cost_so_far[goal]

    def visualize_route(self, path=None):
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
            nx.draw_networkx_edges(self.G, pos, edgelist=path_edges,
                                   edge_color='red', width=4)

        plt.title("Traffic Route Map")
        plt.axis('off')
        plt.show()


def main():
    city = TrafficAwareCityGraph()

    city.train_traffic_model('TrafficDataSet.csv')

    print("\nEnter city connections:")
    num_edges = int(input("Enter number of edges: "))

    for i in range(num_edges):
        print(f"\nEdge {i + 1}")
        from_node = input("From city: ")
        to_node = input("To city: ")
        distance = float(input("Distance (km): "))

        vehicles = {
            'cars': int(input("Number of cars: ")),
            'bikes': int(input("Number of bikes: ")),
            'buses': int(input("Number of buses: ")),
            'trucks': int(input("Number of trucks: "))
        }

        city.add_edge(from_node, to_node, distance, vehicles)

    print("\nRoute Planning:")
    start = input("Enter start city: ")
    goal = input("Enter destination city: ")
    day = int(input("Enter day (1-7, where 1 is Monday): "))
    time = input("Enter time (HH:MM:SS): ")

    path, total_cost = city.a_star_search(start, goal, day, time)

    if path:
        print(f"\nOptimal route: {' -> '.join(path)}")
        print(f"Estimated travel time (including traffic): {total_cost:.2f} units")
        city.visualize_route(path)
    else:
        print("\nNo route found between these cities.")


if __name__ == "__main__":
    main()
