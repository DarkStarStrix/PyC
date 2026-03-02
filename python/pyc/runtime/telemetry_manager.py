"""Manages telemetry data for the Vortex system."""

class TelemetryManager:
    """A simple telemetry manager that collects and stores telemetry data."""

    def __init__(self):
        """Initializes the TelemetryManager."""
        self.data = []

    def record(self, data: dict):
        """Records a new piece of telemetry data."""
        self.data.append(data)

    def get_data(self) -> list:
        """Returns all collected telemetry data."""
        return self.data
