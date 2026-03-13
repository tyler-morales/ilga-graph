"""Object type catalog and schema introspection for the Legislative Ontology."""

from __future__ import annotations

from .links import LinkType
from .objects import (
    OBJECT_TYPE_BILL,
    OBJECT_TYPE_COMMITTEE,
    OBJECT_TYPE_HEARING,
    OBJECT_TYPE_LEGISLATOR,
    OBJECT_TYPE_ORGANIZATION,
    OBJECT_TYPE_VOTE_EVENT,
    OBJECT_TYPE_WITNESS_SLIP,
    BaseOntologyObject,
    BillObject,
    CommitteeObject,
    HearingObject,
    LegislatorObject,
    OrganizationObject,
    VoteEventObject,
    WitnessSlipObject,
)


class ObjectRegistry:
    """Type catalog for ontology object and link types. Used for schema introspection."""

    OBJECT_TYPES: dict[str, type[BaseOntologyObject]] = {
        OBJECT_TYPE_LEGISLATOR: LegislatorObject,
        OBJECT_TYPE_BILL: BillObject,
        OBJECT_TYPE_COMMITTEE: CommitteeObject,
        OBJECT_TYPE_ORGANIZATION: OrganizationObject,
        OBJECT_TYPE_VOTE_EVENT: VoteEventObject,
        OBJECT_TYPE_HEARING: HearingObject,
        OBJECT_TYPE_WITNESS_SLIP: WitnessSlipObject,
    }

    LINK_TYPES = [t.value for t in LinkType]

    @classmethod
    def get_object_class(cls, object_type: str) -> type[BaseOntologyObject] | None:
        return cls.OBJECT_TYPES.get(object_type)

    @classmethod
    def object_types(cls) -> list[str]:
        return list(cls.OBJECT_TYPES.keys())
