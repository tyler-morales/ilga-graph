# Legislative Ontology System (LOS)

The **Legislative Ontology** is the **foundational layer of semantic data** for ILGA Graph. It is a major tenet of the project: all legislative and advocacy data is modeled as **objects** (nouns), **links** (relationships), **actions** (verbs), and **logic** (ML and rules). The application plane (FastAPI, GraphQL, HTMX) consumes this layer via the **Ontology SDK (OSDK)** rather than talking directly to raw tables or caches.

---

## Why ontology

- **Single source of meaning** — Legislators, bills, committees, and organizations are first-class objects with stable identities and typed links. The frontend and API never need to know how data is stored.
- **Traceable actions** — Every call, email, or no-answer is an **OntologyAction**; advocacy history and future ROI-style analytics build on this.
- **Logic in one place** — Scorecards, Moneyball, influence, predictions, and alerts attach to objects through the ontology so reasoning is consistent and discoverable.
- **Digital twin** — The ontology models how the Illinois General Assembly and our advocacy effort actually operate, not how individual systems happen to store data.

---

## Three planes

| Plane | Role | Implementation |
|-------|------|-----------------|
| **Data** | Raw ingestion | Scrapers, JSON cache, SQLite (users/outreach), Parquet (ML). |
| **Ontology** | Mapping and logic | `src/ilga_graph/ontology/`: objects, links, actions, mapping, logic, OSDK. |
| **Application** | Interface | FastAPI, HTMX, GraphQL; all consume the OSDK (or GraphQL ontology queries). |

Data flows into the **Mapping Service**, which hydrates **ontology objects** with **links** and **computed logic**. The **OSDK** exposes the only API the application layer should use for legislative and advocacy semantics.

---

## Nouns (objects)

Defined in `src/ilga_graph/ontology/objects.py`. Every object has `object_id`, `object_type`, and `links: list[ObjectLink]`.

| Object | Purpose | Key links |
|--------|---------|-----------|
| **LegislatorObject** | A member. | Sponsors (bills), committee_membership (committees). |
| **BillObject** | A bill. | Sponsors (legislators), committees, vote_events, witness_slips, hearings. |
| **CommitteeObject** | A committee. | Members (legislators), bills, hearings. |
| **OrganizationObject** | Org from witness slips. | Slips filed, legislators aligned. |
| **VoteEventObject** | A roll-call vote. | Bill, yea/nay/present voters. |
| **HearingObject** | A scheduled hearing. | Committee, bills. |

Computed fields (scorecard, moneyball, influence, prediction, controversy) are attached during hydration so the object is self-contained.

---

## Verbs (actions)

Defined in `src/ilga_graph/ontology/actions.py`. Every user advocacy interaction is an **OntologyAction** (e.g. `call_rep`, `send_email`, `no_answer`). After each successful `/outreach/record`, the router builds an action and calls `state.ontology_sdk.execute_action(action)`. Actions are stored in memory (`state.ontology_actions`); `sdk.action_history(member_id)` returns actions that targeted that legislator. This enables “who was contacted?” and future ROI tracking.

---

## Logic

The **Logic** layer (`ontology/logic.py`) attaches ML and rules to objects and emits **alerts**:

- **LogicMonitor.get_alerts()** — Slip anomalies (ML), high-controversy open bills.
- **Computed on objects** — Scorecard, Moneyball, influence on legislators; prediction and controversy on bills (see Mapping Service).

The OSDK exposes `get_alerts()`, `predict_bill(leg_id)`, and `influence_rank(chamber)` so the app can reason over the ontology without touching raw analytics or ML outputs directly.

---

## Ontology SDK (OSDK)

Single entry point in `src/ilga_graph/ontology/sdk.py`. Created at startup after data and influence are loaded; available as `state.ontology_sdk`.

| Capability | Method |
|------------|--------|
| Objects | `get_legislator(id)`, `get_bill(leg_id)`, `get_committee(code)`, `get_organization(canonical_name)` |
| Search | `search(query, types)` |
| Graph | `linked_objects(obj, link_type)`, `neighbors(obj, depth)` |
| Actions | `execute_action(action)`, `action_history(target_id)` |
| Logic | `predict_bill(leg_id)`, `influence_rank(chamber)`, `get_alerts()` |

Routers and GraphQL should use the OSDK (or the ontology GraphQL queries) for legislative and advocacy semantics instead of reaching into `state.member_lookup`, `state.bills_lookup`, etc., for anything that is conceptually an “object” or “link.”

---

## GraphQL

The API exposes the ontology so clients can traverse links and read action history without knowing the database:

- **`ontology_links(object_id, object_type)`** — Returns links for a legislator, bill, or committee (`target_id`, `target_type`, `link_type`).
- **`ontology_action_history(member_id)`** — Returns actions (call/email/no_answer) that targeted that legislator (in-memory; cleared on restart).

See [GraphQL API](../reference/graphql.md#ontology-queries).

---

## File layout

```
src/ilga_graph/ontology/
  __init__.py    # Exports OSDK, MappingService, objects, links, actions, logic
  objects.py     # BaseOntologyObject, LegislatorObject, BillObject, ...
  links.py       # ObjectLink, LinkType
  actions.py     # OntologyAction, ActionType, ActionResult, outreach_event_to_action
  mapping.py     # MappingService (hydrates app_state → ontology objects)
  logic.py       # LogicMonitor, LogicAlert
  sdk.py         # OntologySDK
  registry.py    # ObjectRegistry (type catalog)
```

---

## Testing the ontology

1. **Start the app** — Startup logs “Ontology SDK initialized.” when the OSDK is built.
2. **GraphQL** — At `/graphql`, run `ontology_links(object_id: "MEMBER_ID", object_type: "legislator")` and `ontology_action_history(member_id: "MEMBER_ID")` (use real IDs from your data).
3. **Python** — `state.ontology_sdk.get_legislator(id)`, `sdk.linked_objects(obj, "sponsors")`, `sdk.get_alerts()`.
4. **Actions** — Record a call or email via the UI, then query `ontology_action_history` for that legislator; after restart, history is empty (in-memory only).

---

## References

- **Plan:** The implementation follows the Legislative Ontology System plan (objects, mapping, OSDK, actions, logic, GraphQL).
- **Hardball spec:** Advocacy behavior (e.g. “making the case,” outreach) still follows `docs/hardball-spec/`; the ontology is the *data and logic* layer that supports that behavior.
- **Canonical content:** Messaging and copy remain under the no-hallucination rule; the ontology does not define narrative content, only structure and semantics.
