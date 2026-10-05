# Server maintenance

## Mark the server inactive before maintenance

A server under maintenance is `INACTIVE`. Inactive servers cannot receive workloads. Publish the change as a fact. Do not treat the status change itself as permission to drop assigned work.

## Relocate workloads before the window

Workloads still assigned to the maintenance server are the ones that must move. Select other operational servers as destinations. Leave workloads that are already on healthy servers where they are.

## Validate the relocation first

Build the relocation with the placement optimizer and evaluate it on a copy. Every workload that must move needs a feasible CPU and memory fit. If any of them cannot be placed, do not start maintenance relocation. Report the plan as infeasible and keep the current assignment.

## Verify after the server is restored

When maintenance ends, a `SERVER_RECOVERY` event records that the server is operational again. Confirm the server can host new workloads only after its status is operational. Confirm existing workloads are still on exactly one server and that CPU and memory limits still hold. Recovery of the server does not by itself move workloads back.
