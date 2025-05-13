# Streamlit Simulation Dashboard - app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import simpy
import random
from collections import defaultdict
from scipy import stats

st.set_page_config(layout="wide")
st.title("Discrete-Event Simulation Dashboard")

# Sidebar inputs
st.sidebar.header("Simulation Settings")
SIM_TIME = st.sidebar.number_input("Simulation Time (minutes)", min_value=10, value=100)
REPLICATIONS = st.sidebar.number_input("Number of Replications", min_value=1, value=3)
NUM_SERVERS = st.sidebar.number_input("Number of Servers", min_value=1, value=2)
RENEGE_TIME = st.sidebar.number_input("Max wait before reneging", min_value=1, value=12)
AGING_THRESHOLD = st.sidebar.number_input("Aging threshold (min)", min_value=1, value=5)
MIN_PRIORITY = st.sidebar.number_input("Minimum priority (0 = highest)", min_value=0, value=0)

# Button to run the simulation
run_simulation = st.sidebar.button("Run Simulation")

# Block 2 - Simulation Logic
if run_simulation:
    st.subheader("Simulation Running...")
    entity_logs_all = []

    entity_types_config = [
        {"name": "Standard", "priority": 2, "service_time": 8, "prob": 0.7},
        {"name": "Express", "priority": 1, "service_time": 4, "prob": 0.3}
    ]

    def get_current_interarrival_time(current_time):
        if current_time < 30:
            return 6
        elif current_time < 60:
            return 3
        else:
            return 5

    def run_single_sim(replication_id):
        env = simpy.Environment()
        server = simpy.PriorityResource(env, capacity=NUM_SERVERS)
        type_metrics = defaultdict(lambda: {"wait": [], "service": [], "total": [], "count": 0})
        entity_logs = []
        aged_entities = 0

        def choose_type():
            r = random.random()
            total = 0
            for t in entity_types_config:
                total += t["prob"]
                if r <= total:
                    return t
            return entity_types_config[-1]

        def entity(env, name, server, ent_type):
            nonlocal aged_entities
            arrival = env.now
            current_priority = ent_type["priority"]
            aged = False
            wait_start = env.now

            log_entry = {
                "Replication": replication_id,
                "EntityID": name,
                "Type": ent_type["name"],
                "ArrivalTime": arrival,
                "UsedAging": False,
                "Reneged": False
            }

            while True:
                with server.request(priority=current_priority) as req:
                    result = yield env.any_of([req, env.timeout(RENEGE_TIME), env.timeout(AGING_THRESHOLD)])
                    now = env.now
                    if req in result:
                        if aged:
                            aged_entities += 1
                            log_entry["UsedAging"] = True
                        start = now
                        service_duration = random.expovariate(1.0 / ent_type["service_time"])
                        yield env.timeout(service_duration)
                        end = env.now
                        type_metrics[ent_type["name"]]["wait"].append(start - arrival)
                        type_metrics[ent_type["name"]]["service"].append(end - start)
                        type_metrics[ent_type["name"]]["total"].append(end - arrival)
                        type_metrics[ent_type["name"]]["count"] += 1
                        log_entry["StartService"] = start
                        log_entry["EndService"] = end
                        break
                    elif now - wait_start >= RENEGE_TIME:
                        log_entry["Reneged"] = True
                        break
                    elif current_priority > MIN_PRIORITY:
                        current_priority -= 1
                        aged = True
                        wait_start = now

            entity_logs.append(log_entry)

        def generator(env, server):
            i = 0
            while True:
                i += 1
                yield env.timeout(random.expovariate(1.0 / get_current_interarrival_time(env.now)))
                ent_type = choose_type()
                env.process(entity(env, f"Entity {i}", server, ent_type))

        env.process(generator(env, server))
        env.run(until=SIM_TIME)

        for log in entity_logs:
            entity_logs_all.append(log)

        result = {"Replication": replication_id, "AgedEntities": aged_entities}
        for label, data in type_metrics.items():
            if data["count"] > 0:
                result[f"{label}_AvgWait"] = np.mean(data["wait"])
                result[f"{label}_AvgService"] = np.mean(data["service"])
                result[f"{label}_AvgTotal"] = np.mean(data["total"])
                result[f"{label}_Count"] = data["count"]
        return result

    results = [run_single_sim(i + 1) for i in range(REPLICATIONS)]
    df = pd.DataFrame(results)
    st.success("Simulation complete.")

    # Additional blocks (3 and 4) would continue here...