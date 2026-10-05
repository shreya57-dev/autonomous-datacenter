# Server failure recovery

## Detect an inactive server

A `SERVER_FAILURE` event names the server that failed. Confirm that server id exists. Treat `ServerStatus.INACTIVE` as unavailable. An inactive server cannot accept new workloads. Do not delete the workload records that were on it before recovery moves them.

## Identify affected workloads

Affected workloads are the workloads still assigned to the failed server. Record each workload id and its current CPU and memory requirements. Workloads on other servers are unaffected and must stay on those servers. An empty failed server has nothing to move.

## Find remaining capacity

Recovery candidates are the other servers that are still operational. Reserve the CPU and memory those servers already use. Do not place a recovery workload onto the failed server. Do not evacuate unaffected workloads to make room.

## Plan recovery under constraints

Ask the placement optimizer for a proposal. Planning CPU is the current requirement when no forecast is available, and `max(current CPU, predicted CPU)` when a forecast exists. The optimizer must respect CPU capacity and memory capacity. The proposal is not an allocation. If any affected workload is left unplaced, the recovery plan is infeasible.

## Validate before execution

Evaluate the proposal on a copy of the data center. Check that every affected workload fits the selected operational server. Do not change the real data center during this check. Reject the plan when the copy would exceed CPU or memory.

## Fail safely when recovery is infeasible

If validation fails, stop in a failed state and raise the error. Leave unaffected servers unchanged. Leave the affected workloads on the failed server rather than applying a partial move. Do not report the workloads as recovered.

## Verify after recovery

After a valid plan is applied, the failed server is inactive and no longer hosts the recovered workloads. Each recovered workload is on exactly one selected operational server. CPU and memory utilization on every server stay within capacity. A failed verification rolls the move back.
