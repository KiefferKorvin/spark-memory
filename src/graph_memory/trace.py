import asyncio

from .models import TraceEvent


class TraceService:
    def __init__(self, repository, query_id):
        self.repository, self.query_id = repository, query_id
        self.sequence = 0
        self.lock = asyncio.Lock()

    async def emit(self, event_type, branch=None, node_id=None, **metadata):
        async with self.lock:
            self.sequence += 1
            event = TraceEvent(query_id=self.query_id, sequence=self.sequence, event_type=event_type,
                node_id=node_id or (branch.node_id if branch else None),
                branch_id=branch.id if branch else None,
                parent_branch_id=branch.parent_branch_id if branch else None,
                information_need_id=branch.information_need_id if branch else None, metadata=metadata)
            await self.repository.record("event", f"{self.query_id}:{self.sequence:08}", event.model_dump())
            return event
