# Workload placement policy

## Accept only operational servers

Place a workload only on a server whose status is operational. Inactive servers are excluded from new assignments. `Server.allocate` rejects a non-operational server before it changes that server's workload list.

## Check CPU and memory together

Feasibility requires both resources. Enough CPU with too little memory is a rejection. Enough memory with too little CPU is a rejection. The check includes workloads already assigned to that server.

## Honor workload priority as a label

Each workload has a non-negative integer priority. Priority describes the workload. It does not override CPU or memory capacity. A high-priority workload that fits nowhere stays unplaced.

## Keep placement deterministic

The same workloads, servers, and solver settings produce the same proposal. The baseline scheduler breaks CPU-utilization ties by earlier position in the server list. The optimizer breaks remaining ties by server index after the peak objective. Do not use an unseeded global random choice for placement.

## Treat the optimizer as the constrained authority

The baseline scheduler is a greedy control group and allocates as it goes. Constrained placement for the agent comes from the OR-Tools proposal. That proposal does not mutate servers. The agent allocates on the real data center only after the proposal passes validation.
