# Resource threshold policy

## Treat a breach as a fact

A `RESOURCE_THRESHOLD_BREACH` event reports that a measured value crossed a limit. The event does not choose a new server and does not move a workload. Observation comes first. Any relocation is a later, validated action.

## Watch CPU utilization

Compare each operational server's CPU utilization with the configured CPU limit. A breach means that server is too busy for the current policy. It does not mean the assignment is automatically illegal. Capacity is still the hard constraint of used CPU against CPU capacity.

## Watch memory pressure

Memory pressure is memory utilization against the configured memory limit. A server can be under its memory capacity and still be over the policy threshold. Do not place additional memory demand on that server until a new plan is validated.

## Watch the simplified temperature reading

Temperature in this simulation is a comparative reading: it rises with CPU utilization and falls with rack cooling. It is not a physical facility model and it is not an energy measurement. A temperature breach is a reason to observe and replan, not a claim about real cooling equipment.

## Trigger replanning without acting immediately

Threshold breaches, workload arrivals, and server failures are reasons to inspect state and request a new placement proposal. Executing that proposal still requires CPU and memory validation on a copy, then verification after the real servers change.
