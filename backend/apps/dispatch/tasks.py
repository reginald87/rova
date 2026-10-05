"""Dispatch Celery tasks.

Three jobs, all idempotent so a retry or an overlapping beat tick is harmless:

* ``dispatch_ride_task``    — one dispatch pass for one ride request.
* ``dispatch_retry_task``   — the delayed second look when a pass found nobody.
* ``expire_offers``         — the sweep that expires lapsed offers and
  reassigns the rides they leave open.

Tasks never raise on a domain-level "no driver yet" outcome: that is a normal
result with a retry attached, and turning it into a task failure would fill the
error log with rides that are simply waiting for a driver to come online.
"""

import logging

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from . import choices
from .services import dispatch_service
from .services.errors import DispatchError

logger = logging.getLogger("apps.dispatch")


@shared_task(name="dispatch.dispatch_ride", bind=True, max_retries=0)
def dispatch_ride_task(
    self, ride_request_id: str, trigger: str = choices.DispatchTrigger.RIDE_REQUEST
):
    """Run one dispatch pass, queueing a retry if nobody was found."""
    return _run_pass(ride_request_id=ride_request_id, trigger=trigger)


@shared_task(name="dispatch.retry_dispatch", bind=True, max_retries=0)
def dispatch_retry_task(self, ride_request_id: str):
    """Re-run dispatch after the configured delay."""
    return _run_pass(ride_request_id=ride_request_id, trigger=choices.DispatchTrigger.MANUAL)


@shared_task(name="dispatch.tasks.expire_offers")
def expire_offers(limit: int = 100) -> dict:
    """Expire lapsed offers and re-dispatch the rides they left open."""
    outcomes = dispatch_service.expire_offers_and_reassign(limit=limit)
    offered = sum(1 for outcome in outcomes if outcome.offered)
    logger.info(
        "dispatch_expired_offers reassigned=%d offered=%d",
        len(outcomes),
        offered,
    )
    return {"reassigned": len(outcomes), "offered": offered}


def _run_pass(*, ride_request_id: str, trigger: str) -> dict:
    try:
        outcome = dispatch_service.dispatch_ride(ride_request_id=ride_request_id, trigger=trigger)
    except DispatchError as exc:
        logger.warning(
            "dispatch_skipped ride_request=%s trigger=%s error=%s",
            ride_request_id,
            trigger,
            exc,
        )
        return {"status": "error", "detail": str(exc)}

    if outcome.skipped:
        return {"status": "skipped"}

    # Narrow on the offer itself rather than `outcome.offered`: the property is
    # defined as `offer is not None`, but a type checker cannot follow that, and
    # the four attribute reads below are not worth four suppressions.
    offer = outcome.offer
    if offer is not None:
        return {
            "status": "offered",
            "offer_id": str(offer.pk),
            "driver_id": str(offer.driver_id),
            "distance_meters": offer.distance_to_pickup_meters,
            "expires_at": offer.expires_at.isoformat(),
        }

    if outcome.exhausted:
        logger.info(
            "dispatch_exhausted ride_request=%s reason=%s attempts=%d",
            ride_request_id,
            outcome.reason,
            outcome.attempt.attempt_number if outcome.attempt else 0,
        )
        return {"status": "exhausted", "reason": outcome.reason}

    if outcome.should_retry:
        delay = int(settings.DISPATCH_RETRY_DELAY_SECONDS)
        dispatch_retry_task.apply_async(args=[ride_request_id], countdown=delay)
        logger.info(
            "dispatch_retry_scheduled ride_request=%s reason=%s in=%ds",
            ride_request_id,
            outcome.reason,
            delay,
        )
        return {"status": "retrying", "reason": outcome.reason, "retry_in_seconds": delay}

    return {"status": "noop"}


@shared_task(name="dispatch.refresh_driver_location")
def refresh_driver_location(driver_id: str, latitude: float, longitude: float):
    """Record a driver's latest position.

    Phase 9 calls this from admin/test paths only; Phase 10 replaces it with the
    realtime ingest pipeline. Kept as a task so the ingest entry point does not
    change shape when it does.
    """
    from .services.location_service import get_location_store

    location = get_location_store().record_position(
        driver_id=driver_id, latitude=latitude, longitude=longitude
    )
    return {"driver_id": str(location.driver_id), "recorded_at": timezone.now().isoformat()}
