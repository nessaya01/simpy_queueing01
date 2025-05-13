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

# Entity type configuration
st.sidebar.header("Entity Types")
num_types = st.sidebar.number_input("Number of entity types", min_value=1, max_value=5, value=2)

entity_types_config = []
total_prob = 0

for i in range(num_types):
    with st.sidebar.expander(f"Entity Type {i+1}"):
        name = st.text_input(f"Name {i+1}", value=f"Type{i+1}", key=f"name_{i}")
        priority = st.number_input(f"Priority (0=highest) {i+1}", min_value=0, value=i, key=f"prio_{i}")
        service_time = st.number_input(f"Avg Service Time {i+1} (minutes)", min_value=0.1, value=6.0 - i*1.5, key=f"st_{i}")
        prob = st.slider(f"Arrival Probability {i+1}", min_value=0.0, max_value=1.0, value=round(1.0/num_types, 2), step=0.01, key=f"prob_{i}")
        entity_types_config.append({"name": name, "priority": int(priority), "service_time": service_time, "prob": prob})
        total_prob += prob

# Normalize probabilities
if total_prob > 0:
    for etype in entity_types_config:
        etype["prob"] /= total_prob

# Visualize arrival mix
st.sidebar.subheader("Arrival Mix Visualization")

if entity_types_config:
    labels = [etype["name"] for etype in entity_types_config]
    values = [etype["prob"] for etype in entity_types_config]

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.barh(labels, values, color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Arrival Mix")
    st.sidebar.pyplot(fig)

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
    if df.empty:
        st.error("No simulation results recorded. Try increasing simulation time or reducing reneging behavior.")
    else:
        st.success("Simulation complete.")

    st.subheader("🔍 Debug: Raw Entity Logs")
    if entity_logs_all:
        st.dataframe(pd.DataFrame(entity_logs_all))
    else:
        st.warning("No entities were created or all entities reneged.")

    # ----------------------------
    # Display Summary Table
    # ----------------------------
    st.subheader("Summary Table (All Replications)")
    st.dataframe(df.style.format(precision=2))

    # ----------------------------
    # Entity-Type Aggregation Table
    # ----------------------------
    st.subheader("Aggregated Metrics by Entity Type")
    type_summary = {}
    for col in df.columns:
        if "_AvgWait" in col or "_AvgService" in col or "_AvgTotal" in col or "_Count" in col:
            label = col.split("_")[0]
            if label not in type_summary:
                type_summary[label] = {"Count": 0, "Avg Wait": [], "Avg Service": [], "Avg Total": []}
            if "_AvgWait" in col:
                type_summary[label]["Avg Wait"] += list(df[col].dropna())
            elif "_AvgService" in col:
                type_summary[label]["Avg Service"] += list(df[col].dropna())
            elif "_AvgTotal" in col:
                type_summary[label]["Avg Total"] += list(df[col].dropna())
            elif "_Count" in col:
                type_summary[label]["Count"] += df[col].sum()

    rows = []
    for label, metrics in type_summary.items():
        rows.append({
            "Type": label,
            "Total Served": metrics["Count"],
            "Avg Wait": round(np.mean(metrics["Avg Wait"]), 2) if metrics["Avg Wait"] else None,
            "Avg Service": round(np.mean(metrics["Avg Service"]), 2) if metrics["Avg Service"] else None,
            "Avg Total": round(np.mean(metrics["Avg Total"]), 2) if metrics["Avg Total"] else None,
        })
    st.dataframe(pd.DataFrame(rows))

    # ----------------------------
    # Plot Metrics per Replication
    # ----------------------------
    st.subheader("Plots: Metrics Across Replications")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df.columns:
        if col.endswith("_AvgWait") or col.endswith("_AvgService") or col.endswith("_AvgTotal"):
            ax.plot(df["Replication"], df[col], label=col, marker='o')
    if "AgedEntities" in df.columns:
        ax.plot(df["Replication"], df["AgedEntities"], label="Aged Entities", linestyle='--', marker='x')

    ax.set_xlabel("Replication")
    ax.set_ylabel("Metric Value")
    ax.set_title("Performance Metrics by Replication")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # Optional CSV Export
    # ----------------------------
    st.subheader("Download Results")
    filebase = st.text_input("Base filename (no extension)", value="simulation_output")

    if st.button("Export to CSV"):
        df.to_csv(f"{filebase}.csv", index=False)
        st.success(f"Saved: {filebase}.csv")

    # ----------------------------
    # Gantt-style Entity Timeline Plot
    # ----------------------------
    st.subheader("Entity-Level Gantt Chart")

    rep_to_animate = st.number_input("Replication number to animate (1 to N)", min_value=1, max_value=REPLICATIONS, value=1)

    if entity_logs_all:
        gantt_df = pd.DataFrame([log for log in entity_logs_all if log["Replication"] == rep_to_animate])
        if not gantt_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {etype: plt.cm.tab10(i) for i, etype in enumerate(gantt_df["Type"].unique())}
            yticks = []
            ylabels = []

            for i, row in gantt_df.iterrows():
                y = i
                yticks.append(y)
                ylabels.append(f"{row['Type']}-{row['EntityID']}")
                if row.get("Reneged"):
                    ax.plot(row["ArrivalTime"], y, 'rx', markersize=8)
                elif pd.notnull(row.get("StartService")) and pd.notnull(row.get("EndService")):
                    ax.broken_barh([(row["StartService"], row["EndService"] - row["StartService"])],
                                   (y - 0.4, 0.8),
                                   facecolors=colors.get(row["Type"], "gray"))

            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=8)
            ax.set_xlabel("Simulation Time")
            ax.set_title(f"Gantt Chart: Replication {rep_to_animate}")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("No data available for the selected replication.")
    else:
        st.warning("Entity log is not available. Run the simulation first.")
