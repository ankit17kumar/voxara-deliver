---
id: ndr_policy
workflow: ndr_recovery
lang: en
---
# NDR (Non-Delivery Report) recovery policy

## When an NDR call is triggered
An NDR call is placed when the courier marks a shipment as undelivered. Common NDR reasons are: customer not available, customer refused, address incomplete, phone unreachable, customer asked to reschedule, and COD amount not ready. The agent calls within 2 hours of the NDR being raised, and only between 9 AM and 9 PM IST.

## Reattempt rules
A shipment gets at most 3 delivery attempts in total. The customer may choose a reattempt date within the next 3 days. Sundays are allowed only for metro pincodes. If the customer asks for a date beyond 3 days, the agent offers the latest allowed date or escalates to a human.

## Refusal handling
If the customer says they refused the order, the agent asks for the reason once, without pressure. If the reason is a product problem, a wrong item or a late delivery, the agent records the reason and offers a reattempt only if the customer agrees. If the customer confirms they do not want the order, the agent marks it RTO (return to origin) and thanks them. The agent must never argue, pressure or repeat the offer more than once.

## Fake attempt complaints
If the customer says the courier never came, did not call, or marked the attempt falsely, the agent apologises, records a "fake attempt" complaint against the courier, and schedules a priority reattempt for the next day.

## Outcome codes
Every NDR call ends with exactly one outcome: REATTEMPT_SCHEDULED, ADDRESS_UPDATED, RTO_CONFIRMED, ESCALATED, CALLBACK_REQUESTED or UNREACHABLE.
