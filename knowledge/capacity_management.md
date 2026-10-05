# Capacity management

## Read available capacity

Available CPU is server CPU capacity minus the CPU already required by assigned workloads. Available memory is computed the same way. Utilization is that used amount divided by capacity. Inactive servers are not available capacity for new placements.

## Enforce capacity constraints

A placement is feasible only when added CPU does not exceed CPU capacity and added memory does not exceed memory capacity. Rounding in the optimizer is conservative: demand rounds up and capacity rounds down. A server that is already over capacity is not a valid planning input.

## Avoid overload

Do not append a workload that `Server.can_host` rejects. Prefer a plan that places feasible workloads before lowering peak CPU. Leaving a workload unplaced is allowed when no operational server can take it. Unplaced is a result, not a silent drop of the workload definition.

## Separate current demand from predicted demand

`Workload.cpu_required` is the current demand and must not be overwritten by a forecast. Predicted CPU is a separate next-step value. Planning demand uses `max(current CPU, predicted CPU)` so a higher forecast reserves more CPU. If the history is too short to forecast, plan with the current CPU only.

## Replan when capacity changes

Replan when a workload arrives, when a server fails, or when a threshold breach says current usage is no longer acceptable. Replanning produces a new proposal. It does not by itself move workloads. Apply a new placement only after validation on a copy.
