class XVAInterfaceError(Exception):
    """Base error for native xva interface."""


class ValidationError(XVAInterfaceError):
    """Input validation failure."""


class MappingError(XVAInterfaceError):
    """Dataclass to runtime mapping failure."""


class EngineRunError(XVAInterfaceError):
    """Runtime execution failure."""


class ConflictError(XVAInterfaceError):
    """Conflict in mixed-source merge."""
