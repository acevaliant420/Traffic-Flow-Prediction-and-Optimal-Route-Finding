import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('TrafficDataSet.csv')
data['Traffic Situation'] = data['Traffic Situation'].replace({'low': 1, 'normal': 2, 'high': 3, 'heavy': 4})
data['Day of the week'] = data['Day of the week'].replace(
    {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})

data['hour'] = pd.to_datetime(data['Time']).dt.hour
data['minute'] = pd.to_datetime(data['Time']).dt.minute
data['AM/PM'] = (data['Time'].str.split().str[1] == 'PM').astype(int)
data = data.drop(columns=['Time'], axis=1)

X = data[
    ['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'hour', 'minute', 'AM/PM']]
y = data['Traffic Situation'].values

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

rfc = RandomForestClassifier(random_state=0)
rfc.fit(train_X, train_y)

rfc_pred = rfc.predict(test_X)
accuracy = accuracy_score(test_y, rfc_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.4f}")


def predict_traffic():
    days_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

    try:
        day = input("Enter the day of the week (e.g., Monday): ")
        if day not in days_map:
            raise ValueError("Invalid day input. Please enter a valid day.")

        time_input = input("Enter the time in HH:MM:SS format (e.g., 14:30:00): ")
        time_parts = time_input.split(':')
        if len(time_parts) != 3 or not all(tp.isdigit() for tp in time_parts) or not (
                0 <= int(time_parts[0]) < 24 and 0 <= int(time_parts[1]) < 60 and 0 <= int(time_parts[2]) < 60):
            raise ValueError("Invalid time format. Please use HH:MM:SS format.")

        hour = int(time_parts[0])
        minute = int(time_parts[1])
        am_pm = 1 if hour >= 12 else 0
        hour = hour % 12

        car_count = int(input("Enter the number of cars: "))
        bike_count = int(input("Enter the number of bikes: "))
        bus_count = int(input("Enter the number of buses: "))
        truck_count = int(input("Enter the number of trucks: "))

    except ValueError as e:
        print(e)
        return

    total_vehicles = car_count + bike_count + bus_count + truck_count
    input_data = pd.DataFrame(
        [[0, days_map[day], car_count, bike_count, bus_count, truck_count, total_vehicles, hour, minute, am_pm]],
        columns=['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'hour',
                 'minute', 'AM/PM'])

    input_data_scaled = sc.transform(input_data)
    prediction = rfc.predict(input_data_scaled)

    traffic_situation = {1: 'low', 2: 'normal', 3: 'high', 4: 'heavy'}
    print(f"The predicted traffic situation is: {traffic_situation[prediction[0]]} ({prediction[0]})")


predict_traffic()
