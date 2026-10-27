# Operations knowledge base (generic guidance; independent of the benchmark scenarios)
# Chunk format:  ## ID | tags: a, b | Title   followed by the text.  Tags are used ONLY to evaluate retrieval relevance.
# NOTE: drafted by the project authors; a domain expert should review or replace it before any paper results are reported.

## KB-01 | tags: leak, isolation | Isolating a suspected burst main
A burst is stopped by closing the valves at both ends of the damaged pipe. Every closed valve cuts supply to the customers that depend on that pipe, so isolation trades lost water for an outage. Close the smallest set of valves that separates the suspected pipe from the supply.

## KB-02 | tags: leak, isolation | Dead ends and single feeds
If the isolated pipe is the only feed to a dead-end area, all customers beyond it lose supply until repair. This is accepted for a confirmed burst but is expensive if the pipe was chosen on weak evidence. Check which customers depend on the pipe before closing it.

## KB-03 | tags: leak, diagnosis | What a leak looks like in telemetry
A leak lowers pressure in its surroundings, most strongly near the leak, and the drop persists instead of following the daily demand pattern. Tank levels fall faster than usual and supply flow rises. Pressure sensors close to the leak show larger drops than distant ones.

## KB-04 | tags: leak, localisation | Locating a leak by sectioning
A leak can be located by closing a section and watching whether the loss signature disappears: pressures recover and tank drain slows if the leak was inside the closed section. Each trial costs an outage for the customers in the section and takes time, so it is worthwhile only for large losses with an uncertain location.

## KB-05 | tags: leak, uncertainty | Acting on an uncertain leak location
When several pipes are almost equally likely, closing many valves at once on weak evidence multiplies the outage and rarely stops the loss. Prefer reversible actions, keep the number of closures small, and consider escalating to an operator if the leading candidates have similar probability.

## KB-06 | tags: leak, urgency | Urgency of large losses
Large bursts empty storage tanks and depress pressure for many customers within hours. Urgency is higher when tank levels are falling and when the estimated loss is a large share of system demand. Small losses justify more time for diagnosis.

## KB-07 | tags: pump | Responding to a pump trip
Compare the observed pump status with the status the controls demand. If a pump that should be running is off, start the standby pump when one exists and check the discharge pressure. Do not restart a tripped pump repeatedly before the cause is known.

## KB-08 | tags: pump, controls | Level-controlled pumps
Many pumps start and stop on tank levels. A pump that stays off while the level it watches is below its start level indicates a fault; a pump that is off inside its normal level band does not. Sampled levels can miss a threshold crossing between two samples.

## KB-09 | tags: pump | Pump speed changes
Raising pump speed raises pressure and also increases leakage. Only variable-speed pumps with a head curve can be adjusted, and speed should stay within the manufacturer range. Speed changes do not replace repair of a failed pump.

## KB-10 | tags: pump | Running pumps in parallel
Starting a second pump increases flow but check the combined operating point against the pump curves and avoid pumping against closed valves. A pump on a flat curve can change flow strongly for small head changes.

## KB-11 | tags: pump | Electrical and mechanical pump faults
A trip can come from power loss, protection relays or mechanical damage. Telemetry alone cannot tell them apart; field confirmation is required before repair crews are dispatched.

## KB-12 | tags: surge, diagnosis | Demand surge versus leak
A demand surge raises consumption over a whole area, follows the daily pattern and resolves when demand falls. A leak persists, is concentrated near one location and does not follow consumption. Pressure drops spread over many sensors point to surge; a persistent local drop points to a leak.

## KB-13 | tags: surge | Responding to a demand surge
Increase supply rather than restricting it: start standby capacity or raise pump output within limits. Closing pipes during a surge reduces service. Surges usually need monitoring and notification, not isolation.

## KB-14 | tags: surge, tank | Protecting storage
Tanks supply the network when pumps cannot keep up. Avoid actions that drain tanks below minimum levels and prefer actions that restore storage after the event.

## KB-15 | tags: surge | Temporary localised demand events
Short local demand events such as fire fighting or main flushing lower pressure nearby for a limited time. They resolve without valve operations; treat them as surges.

## KB-16 | tags: constraints | Minimum service pressure
Customers that remain connected must keep pressure above the regulatory minimum. An action that leaves connected customers below that pressure is not acceptable even if it saves water.

## KB-17 | tags: constraints | Over-pressure
Pressure well above a node's usual daily maximum can break old mains and joints. Be cautious with actions that raise pressure: pump speed increases, starting extra pumps or closing pipes that redirect flow.

## KB-18 | tags: constraints, valves | Valve operation limits
Operate a limited number of valves at a time and record every operation. Valves closed for an incident must be reopened after repair so customers are restored.

## KB-19 | tags: constraints | Critical customers
Hospitals, dialysis centres and similar customers must be notified before any outage affecting them. Avoid isolating pipes that feed critical customers unless the loss is severe.

## KB-20 | tags: constraints, verification | Verifying actions before execution
When time permits, check a planned action with a hydraulic model before executing it. The model confirms pressures, storage and outage effects; it is only as good as its calibration, so compare predictions with telemetry after the action.

## KB-21 | tags: uncertainty, escalation | When to escalate to an operator
Escalate when the evidence is ambiguous, when two explanations suggest opposite actions, or when the action would cut off many customers on weak evidence. Escalation is a legitimate outcome, not a failure.

## KB-22 | tags: uncertainty | Reporting confidence
State how likely each explanation is and what would change the recommendation. Do not present a ranked guess as a confirmed diagnosis.

## KB-23 | tags: uncertainty, reversibility | Prefer reversible actions
When the diagnosis is uncertain, prefer actions that can be undone quickly and whose cost is small if they turn out to be wrong, over irreversible or wide-reaching ones.

## KB-24 | tags: diagnosis | Alarm latency
Telemetry is available only after an alarm and sensor noise delays recognition. Early recommendations are based on a few samples; recommendations should be revisited as more data arrive.

## KB-25 | tags: communication | Customer notification
Notify customers affected by planned or emergency outages and give an expected restoration time. Keep notifications factual.

## KB-26 | tags: communication | Incident log
Record the alarm time, the evidence, the actions taken and the reasons. The log supports later analysis and regulators' enquiries.

## KB-27 | tags: distractor | Chlorine residual monitoring
Free chlorine residual should be monitored at representative points in the network. Residual falls with water age and temperature; low residual triggers flushing or booster dosing.

## KB-28 | tags: distractor | Unidirectional flushing
Flushing removes sediment and biofilm by sending high velocity flow in one direction through a sequence of mains. It is scheduled in low-demand periods and needs notification.

## KB-29 | tags: distractor | Customer meter reading
Meters are read on a fixed cycle. Estimated readings should be flagged and replaced by actual readings at the next visit.

## KB-30 | tags: distractor | Pipe materials and corrosion
Cast iron mains corrode internally and tuberculate, which raises roughness and lowers capacity. Plastic mains resist corrosion but are sensitive to impact and pressure surges.

## KB-31 | tags: distractor | Procurement of valves and fittings
Purchase isolation valves from approved suppliers, keep strategic spares and check certification of materials in contact with drinking water.

## KB-32 | tags: distractor | Seasonal demand planning
Summer demand rises with irrigation and temperature. Capacity reviews should compare peak-day demand with pump and storage capacity every year.

## KB-33 | tags: distractor | GIS data quality
Asset records should be reconciled with field surveys regularly. Missing valves or wrong diameters in the GIS cause errors in hydraulic models.

## KB-34 | tags: distractor | Billing disputes
Disputed bills with unusually high consumption may indicate private-side leaks; the utility can offer a leak adjustment policy.
