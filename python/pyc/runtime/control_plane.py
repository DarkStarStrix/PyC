"""Defines the main control plane for the Vortex system."""

from nexa_vortex.core import CpuDispatcher

class ControlPlane:
    """The main control plane for the Vortex system."""

    def __init__(self, dispatcher: CpuDispatcher):
        """
        Initializes the ControlPlane.

        Args:
            dispatcher: The CPU dispatcher for executing commands.
        """
        self.dispatcher = dispatcher
        print("Control Plane Initialized")

    def send_command(self, command: str):
        """Sends a command to the Vortex system for execution."""
        print(f"Dispatching command: {command}")
        # This is a simplified example. In a real scenario, the command
        # would be serialized and processed by a worker.
        self.dispatcher.dispatch(lambda: print(f"Executing command: {command}"))
