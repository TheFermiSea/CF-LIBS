"""
Example usage of hardware interfaces.

This demonstrates how to use the hardware interfaces for CF-LIBS,
including spectrographs, lasers, motion stages, and flow regulators.
"""

from pathlib import Path
from cflibs.hardware import (
    HardwareManager,
    SpectrographHardware,
    LaserHardware,
    MotionStageHardware,
    FlowRegulatorHardware
)
from cflibs.core.logging_config import setup_logging

# Setup logging
setup_logging()


def example_spectrograph():
    """Example: Using spectrograph hardware."""
    print("\n=== Spectrograph Example ===")
    
    spec = SpectrographHardware(
        name="main_spectrograph",
        config={
            'model': 'Andor iDus',
            'resolution_nm': 0.1,
            'wavelength_range': (200.0, 800.0),
            'pixels': 2048
        }
    )
    
    # Use context manager for automatic connection/disconnection
    with spec:
        # Set exposure time
        spec.set_exposure_time(100.0)  # ms
        print(f"Exposure time: {spec.get_exposure_time()} ms")
        
        # Set gain
        spec.set_gain(1.0)
        
        # Acquire spectrum
        wavelength, intensity = spec.acquire_spectrum()
        print(f"Acquired spectrum: {len(wavelength)} points")
        print(f"Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} nm")
        
        # Get status
        status = spec.get_status()
        print(f"Status: {status['status']}")
        print(f"Model: {status['model']}")


def example_laser():
    """Example: Using laser hardware."""
    print("\n=== Laser Example ===")
    
    laser = LaserHardware(
        name="main_laser",
        config={
            'model': 'Nd:YAG',
            'wavelength_nm': 1064.0,
            'max_power_mW': 1000.0,
            'max_energy_mJ': 100.0
        }
    )
    
    with laser:
        # Enable laser
        laser.enable()
        print("Laser enabled")
        
        # Set pulse parameters
        laser.set_pulse_energy(50.0)  # mJ
        laser.set_repetition_rate(10.0)  # Hz
        print(f"Pulse energy: {laser.get_power()} mW")
        
        # Fire laser
        laser.fire(n_pulses=1)
        print("Laser fired")
        
        # Get laser info
        info = laser.get_laser_info()
        print(f"Laser wavelength: {info['wavelength_nm']} nm")
        print(f"Max power: {info['max_power_mW']} mW")


def example_motion_stage():
    """Example: Using motion stage hardware."""
    print("\n=== Motion Stage Example ===")
    
    stage = MotionStageHardware(
        name="xyz_stage",
        config={
            'axes': ['X', 'Y', 'Z'],
            'travel_range': {
                'X': (0.0, 100.0),
                'Y': (0.0, 100.0),
                'Z': (0.0, 50.0)
            }
        }
    )
    
    with stage:
        # Home all axes
        stage.home()
        print("Axes homed")
        
        # Move to position
        stage.move_to('X', 50.0, wait=True)
        print(f"X position: {stage.get_position('X')} mm")
        
        # Relative movement
        stage.move_relative('Y', 10.0)
        print(f"Y position: {stage.get_position('Y')} mm")
        
        # Set velocity
        stage.set_velocity('X', 25.0)  # mm/s
        print("Velocity set")
        
        # Get available axes
        axes = stage.get_available_axes()
        print(f"Available axes: {axes}")


def example_flow_regulator():
    """Example: Using flow regulator hardware."""
    print("\n=== Flow Regulator Example ===")
    
    flow = FlowRegulatorHardware(
        name="powder_feeder",
        config={
            'model': 'Vibra Screw',
            'max_flow_rate_g_s': 10.0,
            'min_flow_rate_g_s': 0.01
        }
    )
    
    with flow:
        # Set flow rate
        flow.set_flow_rate(5.0)  # g/s
        print(f"Flow rate: {flow.get_flow_rate()} g/s")
        
        # Open valve and start flow
        flow.open_valve()
        flow.start_flow()
        print("Flow started")
        
        # Monitor powder level
        level = flow.get_powder_level()
        print(f"Powder level: {level} g")
        
        # Stop flow
        flow.stop_flow()
        flow.close_valve()
        print("Flow stopped")


def example_hardware_manager():
    """Example: Using hardware manager for coordinated control."""
    print("\n=== Hardware Manager Example ===")
    
    # Create manager from config file
    config_path = Path(__file__).parent / "hardware_config_example.yaml"
    
    if config_path.exists():
        manager = HardwareManager(config_path=config_path)
        
        with manager:
            # Get status of all components
            status = manager.get_all_status()
            print(f"Loaded {len(status)} components:")
            for name, comp_status in status.items():
                print(f"  - {name}: {comp_status['status']}")
            
            # Get ready components
            ready = manager.get_ready_components()
            print(f"\nReady components: {ready}")
            
            # Access specific components
            laser = manager.get_component("main_laser")
            spec = manager.get_component("main_spectrograph")
            
            if laser and spec:
                # Coordinated operation
                print("\nCoordinated operation:")
                laser.enable()
                laser.fire()
                wavelength, intensity = spec.acquire_spectrum()
                print(f"Acquired spectrum after laser fire: {len(intensity)} points")
    else:
        print(f"Config file not found: {config_path}")
        print("Creating components manually...")
        
        # Create manager and add components manually
        manager = HardwareManager()
        
        manager.add_component("laser", LaserHardware(name="laser"))
        manager.add_component("spectrograph", SpectrographHardware(name="spectrograph"))
        
        with manager:
            status = manager.get_all_status()
            print(f"Components: {list(status.keys())}")


if __name__ == "__main__":
    print("CF-LIBS Hardware Interface Examples")
    print("=" * 50)
    
    # Run examples
    example_spectrograph()
    example_laser()
    example_motion_stage()
    example_flow_regulator()
    example_hardware_manager()
    
    print("\n" + "=" * 50)
    print("Examples complete!")

