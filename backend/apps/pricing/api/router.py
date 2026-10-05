"""Pricing API.

Public (any authenticated user):
  - GET  /api/v1/pricing/vehicle-categories  — the vehicle category catalog
  - POST /api/v1/pricing/estimate            — fare estimate (server-computed)

Administrative (ADMIN / OPERATIONS roles only, every mutation audited):
  - GET/POST        /api/v1/pricing/rules
  - POST            /api/v1/pricing/rules/{rule_id}/versions
  - POST            /api/v1/pricing/rules/versions/{version_id}/publish
  - GET/POST        /api/v1/pricing/commissions
  - POST            /api/v1/pricing/commissions/{commission_id}/versions
  - POST            /api/v1/pricing/commissions/versions/{version_id}/publish
"""

from datetime import UTC

from django.http import HttpRequest
from ninja import Router
from ninja.errors import HttpError

from apps.accounts.auth import require_authenticated_user, require_role
from apps.accounts.services.audit import record
from apps.geography.api.schemas import CoordinateSchema
from apps.geography.services.geography_service import InvalidCoordinateError
from apps.vehicles.services.vehicle_service import get_vehicle_categories

from ..services.pricing_service import (
    CommissionRuleNotFoundError,
    CommissionRuleVersionNotFoundError,
    CurrencyNotSupportedError,
    InvalidCancellationInputError,
    InvalidCommissionConfigurationError,
    InvalidFareInputError,
    InvalidPricingConfigurationError,
    NoApplicableCommissionRuleError,
    NoApplicablePricingRuleError,
    PricingConfigurationConflictError,
    PricingError,
    PricingRuleNotFoundError,
    PricingRuleVersionNotFoundError,
    PricingUnavailableError,
    UnsupportedVehicleCategoryError,
    create_commission_rule,
    create_commission_rule_version,
    create_pricing_rule,
    create_pricing_rule_version,
    estimate_fare,
    get_latest_commission_version,
    get_latest_pricing_version,
    list_commission_rules,
    list_pricing_rules,
    publish_commission_rule_version,
    publish_pricing_rule_version,
)
from .schemas import (
    CategoryRefSchema,
    CommissionRuleCreateRequest,
    CommissionRuleSummaryResponse,
    CommissionRuleVersionCreateRequest,
    CommissionRuleVersionResponse,
    FareComponentSchema,
    FareEstimateRequest,
    FareEstimateResponse,
    FareLocationInfoSchema,
    PricingRuleCreateRequest,
    PricingRuleInfoSchema,
    PricingRuleSummaryResponse,
    PricingRuleVersionCreateRequest,
    PricingRuleVersionResponse,
    VehicleCategoryRefSchema,
)

router = Router(tags=["pricing"])

_ANY_AUTH = require_authenticated_user()
_ADMIN_AUTH = require_role("ADMIN", "OPERATIONS")

_ERROR_MAP: dict[type, tuple[int, str]] = {
    InvalidCoordinateError: (400, "INVALID_COORDINATE"),
    UnsupportedVehicleCategoryError: (400, "UNSUPPORTED_VEHICLE_CATEGORY"),
    CurrencyNotSupportedError: (400, "UNSUPPORTED_CURRENCY"),
    InvalidFareInputError: (400, "INVALID_FARE_INPUT"),
    InvalidCancellationInputError: (400, "INVALID_CANCELLATION_INPUT"),
    PricingUnavailableError: (400, "PRICING_UNAVAILABLE"),
    NoApplicablePricingRuleError: (400, "NO_APPLICABLE_PRICING_RULE"),
    NoApplicableCommissionRuleError: (400, "NO_APPLICABLE_COMMISSION_RULE"),
    InvalidPricingConfigurationError: (400, "INVALID_PRICING_CONFIGURATION"),
    InvalidCommissionConfigurationError: (400, "INVALID_COMMISSION_CONFIGURATION"),
    PricingRuleNotFoundError: (404, "PRICING_RULE_NOT_FOUND"),
    CommissionRuleNotFoundError: (404, "COMMISSION_RULE_NOT_FOUND"),
    PricingRuleVersionNotFoundError: (404, "PRICING_RULE_VERSION_NOT_FOUND"),
    CommissionRuleVersionNotFoundError: (404, "COMMISSION_RULE_VERSION_NOT_FOUND"),
    PricingConfigurationConflictError: (409, "PRICING_CONFIGURATION_CONFLICT"),
}


def _http_error(exc: Exception) -> HttpError:
    status, code = _ERROR_MAP.get(type(exc), (400, type(exc).__name__))
    return HttpError(status, f"{code}: {exc}")


def _audit(request: HttpRequest, action: str, details: dict | None = None) -> None:
    user = getattr(request, "auth", None)
    record(
        action=action,
        user_id=str(user.pk) if user is not None else None,
        request_id=getattr(request, "id", None),
        ip=request.META.get("REMOTE_ADDR"),
        success=True,
        details=details,
    )


def _coords(latitude, longitude) -> CoordinateSchema:
    return CoordinateSchema(latitude=float(latitude), longitude=float(longitude))


def _money(value) -> float:
    return float(value)


def _category_ref(category) -> CategoryRefSchema:
    return CategoryRefSchema(key=category.key, name=category.name)


def _pricing_rule_info(rule, version) -> PricingRuleInfoSchema:
    return PricingRuleInfoSchema(
        code=rule.code,
        version=version,
        scope_type=rule.scope_type,
        priority=rule.priority,
        city_id=str(rule.city_id) if rule.city_id else None,
        city_name=rule.city.name if rule.city_id else None,
        service_zone_id=str(rule.service_zone_id) if rule.service_zone_id else None,
        service_zone_name=rule.service_zone.name if rule.service_zone_id else None,
    )


def _fare_response(fare) -> FareEstimateResponse:
    components = fare.components or {}
    location = FareLocationInfoSchema(
        city_id=str(fare.city_id) if fare.city_id else None,
        city_name=fare.city.name if fare.city_id else None,
        service_zone_id=str(fare.service_zone_id) if fare.service_zone_id else None,
        service_zone_name=fare.service_zone.name if fare.service_zone_id else None,
        service_available=fare.service_zone_id is not None,
    )
    rule_info = None
    if fare.pricing_rule_id is not None:
        rule_info = PricingRuleInfoSchema(
            code=fare.pricing_rule_code,
            version=fare.pricing_rule_version,
            scope_type=fare.pricing_rule_snapshot.get("scope_type", "GLOBAL"),
            priority=int(fare.pricing_rule_snapshot.get("priority", 0)),
            city_id=fare.pricing_rule_snapshot.get("city_id"),
            city_name=fare.city.name if fare.city_id else None,
            service_zone_id=fare.pricing_rule_snapshot.get("service_zone_id"),
            service_zone_name=fare.service_zone.name if fare.service_zone_id else None,
        )
    return FareEstimateResponse(
        id=str(fare.pk),
        calculation_type=fare.calculation_type,
        currency=fare.currency,
        category=_category_ref(fare.vehicle_category),
        origin=_coords(fare.origin_latitude, fare.origin_longitude),
        destination=_coords(fare.destination_latitude, fare.destination_longitude),
        distance_km=_money(components.get("distance_km", "0")),
        duration_minutes=_money(components.get("duration_minutes", "0")),
        location=location,
        pricing_rule=rule_info,
        components=FareComponentSchema(
            base_fare=_money(components.get("base_fare", "0")),
            distance_charge=_money(components.get("distance_charge", "0")),
            time_charge=_money(components.get("time_charge", "0")),
            subtotal=_money(components.get("subtotal", "0")),
            surge_multiplier=_money(components.get("surge_multiplier", "1.00")),
            surge_adjustment=_money(components.get("surge_adjustment", "0")),
            service_fee=_money(components.get("service_fee", "0")),
            booking_fee=_money(components.get("booking_fee", "0")),
            gross=_money(components.get("gross", "0")),
            minimum_fare=_money(components.get("minimum_fare", "0")),
            minimum_fare_applied=bool(components.get("minimum_fare_applied", False)),
            discount=_money(components.get("discount", "0")),
            total=_money(components.get("total", "0")),
        ),
        created_at=fare.created_at,
    )


def _coerce_aware(value):
    """Return an aware datetime (naive inputs are assumed to be UTC).

    Uses ``datetime.UTC``: ``django.utils.timezone.utc`` was removed in Django 5.0,
    so an admin posting a rule window without an offset would otherwise raise
    ``AttributeError`` — a 500 on a valid request rather than the stored value the
    caller asked for.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _pricing_version_response(version) -> PricingRuleVersionResponse:
    return PricingRuleVersionResponse(
        id=str(version.pk),
        version=version.version,
        status=version.status,
        effective_from=version.effective_from,
        effective_to=version.effective_to,
        base_fare=_money(version.base_fare),
        price_per_km=_money(version.price_per_km),
        price_per_minute=_money(version.price_per_minute),
        service_fee=_money(version.service_fee),
        booking_fee=_money(version.booking_fee),
        minimum_fare=_money(version.minimum_fare),
        surge_multiplier=_money(version.surge_multiplier),
        free_cancellation_seconds=version.free_cancellation_seconds,
        cancellation_fee=_money(version.cancellation_fee),
    )


def _pricing_rule_response(rule) -> PricingRuleSummaryResponse:
    latest = get_latest_pricing_version(rule)
    return PricingRuleSummaryResponse(
        id=str(rule.pk),
        code=rule.code,
        name=rule.name,
        description=rule.description,
        scope_type=rule.scope_type,
        city_id=str(rule.city_id) if rule.city_id else None,
        city_name=rule.city.name if rule.city_id else None,
        service_zone_id=str(rule.service_zone_id) if rule.service_zone_id else None,
        service_zone_name=rule.service_zone.name if rule.service_zone_id else None,
        vehicle_category_key=rule.vehicle_category.key,
        currency=rule.currency,
        priority=rule.priority,
        is_active=rule.is_active,
        latest_version=_pricing_version_response(latest) if latest else None,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


def _commission_version_response(version) -> CommissionRuleVersionResponse:
    return CommissionRuleVersionResponse(
        id=str(version.pk),
        version=version.version,
        status=version.status,
        mode=version.mode,
        value=_money(version.value),
        min_commission=(
            _money(version.min_commission) if version.min_commission is not None else None
        ),
        max_commission=(
            _money(version.max_commission) if version.max_commission is not None else None
        ),
        effective_from=version.effective_from,
        effective_to=version.effective_to,
    )


def _commission_rule_response(rule) -> CommissionRuleSummaryResponse:
    latest = get_latest_commission_version(rule)
    return CommissionRuleSummaryResponse(
        id=str(rule.pk),
        code=rule.code,
        name=rule.name,
        description=rule.description,
        scope_type=rule.scope_type,
        city_id=str(rule.city_id) if rule.city_id else None,
        city_name=rule.city.name if rule.city_id else None,
        service_zone_id=str(rule.service_zone_id) if rule.service_zone_id else None,
        service_zone_name=rule.service_zone.name if rule.service_zone_id else None,
        vehicle_category_key=rule.vehicle_category.key if rule.vehicle_category_id else None,
        currency=rule.currency,
        priority=rule.priority,
        is_active=rule.is_active,
        latest_version=_commission_version_response(latest) if latest else None,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/pricing/vehicle-categories",
    response=list[VehicleCategoryRefSchema],
    auth=_ANY_AUTH,
    url_name="pricing_vehicle_categories",
    summary="List the active vehicle categories.",
    description="Reuses the Phase 5 vehicle category catalog — the canonical source.",
)
def pricing_vehicle_categories(request: HttpRequest):
    return [
        VehicleCategoryRefSchema(
            id=str(c.pk),
            key=c.key,
            name=c.name,
            description=c.description,
            sort_order=c.sort_order,
            is_active=c.is_active,
        )
        for c in get_vehicle_categories()
    ]


@router.post(
    "/pricing/estimate",
    response=FareEstimateResponse,
    auth=_ANY_AUTH,
    url_name="pricing_estimate",
    summary="Estimate a fare for a route and vehicle category.",
    description=(
        "The server resolves the route, selects the applicable pricing rule "
        "(service zone > city > global), and computes distance via PostGIS. The "
        "result is persisted and reproducible. The client may only supply an "
        "estimated duration; no client-supplied prices are ever trusted."
    ),
)
def pricing_estimate(request: HttpRequest, payload: FareEstimateRequest):
    try:
        fare = estimate_fare(
            origin_lat=payload.origin.latitude,
            origin_lon=payload.origin.longitude,
            destination_lat=payload.destination.latitude,
            destination_lon=payload.destination.longitude,
            vehicle_category_key=payload.category,
            currency=payload.currency,
            estimated_duration_seconds=payload.estimated_duration_seconds,
        )
    except (PricingError, InvalidCoordinateError) as exc:
        raise _http_error(exc) from exc
    return _fare_response(fare)


# ---------------------------------------------------------------------------
# Administrative: pricing rules
# ---------------------------------------------------------------------------


@router.get(
    "/pricing/rules",
    response=list[PricingRuleSummaryResponse],
    auth=_ADMIN_AUTH,
    url_name="pricing_rules_list",
    summary="List pricing rules (administrative).",
)
def pricing_rules_list(
    request: HttpRequest,
    scope_type: str | None = None,
    city_id: str | None = None,
    category: str | None = None,
    status: str | None = None,
):
    return [
        _pricing_rule_response(rule)
        for rule in list_pricing_rules(
            scope_type=scope_type, city_id=city_id, category_key=category, status=status
        )
    ]


@router.post(
    "/pricing/rules",
    response=PricingRuleSummaryResponse,
    auth=_ADMIN_AUTH,
    url_name="pricing_rules_create",
    summary="Create a pricing rule (administrative).",
)
def pricing_rules_create(request: HttpRequest, payload: PricingRuleCreateRequest):
    try:
        rule = create_pricing_rule(
            code=payload.code,
            name=payload.name,
            scope_type=payload.scope_type,
            vehicle_category_key=payload.vehicle_category_key,
            currency=payload.currency,
            city_id=payload.city_id,
            service_zone_id=payload.service_zone_id,
            priority=payload.priority,
            description=payload.description,
        )
    except PricingError as exc:
        raise _http_error(exc) from exc
    _audit(request, "pricing.rule.create", details={"rule_code": rule.code})
    return _pricing_rule_response(rule)


@router.post(
    "/pricing/rules/{rule_id}/versions",
    response=PricingRuleVersionResponse,
    auth=_ADMIN_AUTH,
    url_name="pricing_rule_versions_create",
    summary="Add a new DRAFT version to a pricing rule (administrative).",
)
def pricing_rule_versions_create(
    request: HttpRequest, rule_id: str, payload: PricingRuleVersionCreateRequest
):
    try:
        version = create_pricing_rule_version(
            rule_id=rule_id,
            base_fare=payload.base_fare,
            price_per_km=payload.price_per_km,
            price_per_minute=payload.price_per_minute,
            service_fee=payload.service_fee,
            booking_fee=payload.booking_fee,
            minimum_fare=payload.minimum_fare,
            surge_multiplier=payload.surge_multiplier,
            free_cancellation_seconds=payload.free_cancellation_seconds,
            cancellation_fee=payload.cancellation_fee,
            effective_from=_coerce_aware(payload.effective_from),
            effective_to=_coerce_aware(payload.effective_to),
        )
    except PricingError as exc:
        raise _http_error(exc) from exc
    _audit(request, "pricing.rule.version.create", details={"version_id": str(version.pk)})
    return _pricing_version_response(version)


@router.post(
    "/pricing/rules/versions/{version_id}/publish",
    response=PricingRuleVersionResponse,
    auth=_ADMIN_AUTH,
    url_name="pricing_rule_version_publish",
    summary="Publish a pricing rule version, making it effective (administrative).",
    description=(
        "Publishing validates the configuration and rejects overlapping or "
        "ambiguous windows inside a transaction with row locks. Published "
        "versions are immutable."
    ),
)
def pricing_rule_version_publish(request: HttpRequest, version_id: str):
    try:
        version = publish_pricing_rule_version(version_id)
    except PricingError as exc:
        raise _http_error(exc) from exc
    _audit(
        request,
        "pricing.rule.version.publish",
        details={"version_id": str(version.pk), "rule_code": version.rule.code},
    )
    return _pricing_version_response(version)


# ---------------------------------------------------------------------------
# Administrative: commission rules
# ---------------------------------------------------------------------------


@router.get(
    "/pricing/commissions",
    response=list[CommissionRuleSummaryResponse],
    auth=_ADMIN_AUTH,
    url_name="pricing_commissions_list",
    summary="List commission rules (administrative).",
)
def pricing_commissions_list(
    request: HttpRequest,
    scope_type: str | None = None,
    city_id: str | None = None,
    category: str | None = None,
    status: str | None = None,
):
    return [
        _commission_rule_response(rule)
        for rule in list_commission_rules(
            scope_type=scope_type, city_id=city_id, category_key=category, status=status
        )
    ]


@router.post(
    "/pricing/commissions",
    response=CommissionRuleSummaryResponse,
    auth=_ADMIN_AUTH,
    url_name="pricing_commissions_create",
    summary="Create a commission rule (administrative).",
)
def pricing_commissions_create(request: HttpRequest, payload: CommissionRuleCreateRequest):
    try:
        rule = create_commission_rule(
            code=payload.code,
            name=payload.name,
            scope_type=payload.scope_type,
            currency=payload.currency,
            vehicle_category_key=payload.vehicle_category_key,
            city_id=payload.city_id,
            service_zone_id=payload.service_zone_id,
            priority=payload.priority,
            description=payload.description,
        )
    except PricingError as exc:
        raise _http_error(exc) from exc
    _audit(request, "pricing.commission.create", details={"rule_code": rule.code})
    return _commission_rule_response(rule)


@router.post(
    "/pricing/commissions/{commission_id}/versions",
    response=CommissionRuleVersionResponse,
    auth=_ADMIN_AUTH,
    url_name="pricing_commission_versions_create",
    summary="Add a new DRAFT version to a commission rule (administrative).",
)
def pricing_commission_versions_create(
    request: HttpRequest, commission_id: str, payload: CommissionRuleVersionCreateRequest
):
    try:
        version = create_commission_rule_version(
            rule_id=commission_id,
            mode=payload.mode,
            value=payload.value,
            min_commission=payload.min_commission,
            max_commission=payload.max_commission,
            effective_from=_coerce_aware(payload.effective_from),
            effective_to=_coerce_aware(payload.effective_to),
        )
    except PricingError as exc:
        raise _http_error(exc) from exc
    _audit(request, "pricing.commission.version.create", details={"version_id": str(version.pk)})
    return _commission_version_response(version)


@router.post(
    "/pricing/commissions/versions/{version_id}/publish",
    response=CommissionRuleVersionResponse,
    auth=_ADMIN_AUTH,
    url_name="pricing_commission_version_publish",
    summary="Publish a commission rule version (administrative).",
)
def pricing_commission_version_publish(request: HttpRequest, version_id: str):
    try:
        version = publish_commission_rule_version(version_id)
    except PricingError as exc:
        raise _http_error(exc) from exc
    _audit(
        request,
        "pricing.commission.version.publish",
        details={"version_id": str(version.pk), "rule_code": version.rule.code},
    )
    return _commission_version_response(version)
