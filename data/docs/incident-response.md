# Aurora Cloud Incident Response Runbook

## Severity levels

Severity 1 means a complete outage of a production service or confirmed data loss. Severity 2 means major degradation affecting more than 20% of requests in a region. Severity 3 means a minor issue with a workaround. Severity 4 is a cosmetic issue or a question.

## Response targets

For Severity 1, the on-call engineer must acknowledge the page within 5 minutes and an incident commander must be assigned within 15 minutes. For Severity 2, acknowledgement is required within 15 minutes. Severity 3 and 4 issues are handled during business hours, with a target of the next business day.

If a Severity 1 page is not acknowledged within 5 minutes, it escalates automatically to the secondary on-call, and after 10 minutes to the engineering manager.

## Communication

During a Severity 1 incident the status page at status.auroracloud.example must be updated every 30 minutes until the incident is resolved. During Severity 2 incidents updates are posted every 60 minutes. The incident commander owns all external communication; engineers working the incident do not post to customers directly.

## Postmortems

A blameless postmortem is required for every Severity 1 and Severity 2 incident. The postmortem draft is due within 5 business days of resolution and is reviewed in the weekly reliability meeting held on Thursdays. Action items from postmortems are tracked in the RELY project and must have an owner and a due date.
