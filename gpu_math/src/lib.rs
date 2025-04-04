mod gpu_utils;

// error modules
mod errors;
use errors::GpuMathNotInitializedError;

// Publically exposed modules
pub mod math;

// Include these when the library is imported
pub use math::*;

// Library inner imports
use matrix::MatrixPipelines;
use pollster::FutureExt;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

static mut GPU_MATHS: Option<GpuMath> = None;

struct GpuMath {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,

    // Math Pipelines
    matrix_pipelines: MatrixPipelines,
}

/// This initializes all the required gpu devices for the gpu math library
/// This must be called before anything in this library is used
pub fn init() {
    // Initialize all the gpu required values
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .block_on()
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("Device and Queue"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                ..Default::default()
            },
            None,
        )
        .block_on()
        .unwrap();

    // Create the pipelines for matrix math
    let matrix_pipelines = unsafe { MatrixPipelines::init().unwrap_unchecked() };

    unsafe {
        GPU_MATHS = Some(GpuMath {
            instance,
            adapter,
            device,
            queue,
            matrix_pipelines,
        });
    }
}

fn test_init(location: &str) -> Result<(), GpuMathNotInitializedError> {
    unsafe {
        #[allow(static_mut_refs)]
        if GPU_MATHS.is_none() {
            Err(GpuMathNotInitializedError(location.to_string()))
        } else {
            Ok(())
        }
    }
}

// IMPORTANT: make sure to call test_init before this function
unsafe fn get_device<'a>() -> &'a Device {
    #[allow(static_mut_refs)]
    unsafe {
        &GPU_MATHS.as_ref().unwrap_unchecked().device
    }
}

// IMPORTANT: make sure to call test_init before this function
unsafe fn get_queue<'a>() -> &'a Queue {
    #[allow(static_mut_refs)]
    unsafe {
        &GPU_MATHS.as_ref().unwrap_unchecked().queue
    }
}
