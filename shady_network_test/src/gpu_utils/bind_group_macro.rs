pub use wgpu::BindGroupDescriptor;
pub use wgpu::BindGroupEntry;
pub use wgpu::BindGroupLayoutDescriptor;
pub use wgpu::BindGroupLayoutEntry;
pub use wgpu::BindingType;
pub use wgpu::BufferBindingType;
pub use wgpu::ShaderStages;

pub enum Bbt {
    Uniform,
    Storage { read_only: bool },
}

/// Macro for creating bind groups
///
/// # Arguments
///
/// * `device` - reference to wgpu device for creating bind groups
/// * `label` - name of the bind group
/// * `(binding, buffer, type)...` - tuples containing the binding number, the buffer to be binded, and the type of binding (i.e. Storage or Uniform)
///
/// # Returns
///
/// `(BindGroupLayout, BindGroup)` tuple containing the resulting bind group information
#[macro_export]
macro_rules! create_buffer_bind_group {
    ( $device:expr, $label:expr, $( ($binding:expr, $buffer:expr, $type:expr) ),* ) => {
        {
            use crate::gpu_utils::bind_group_macro::*;

            let mut layout_entries = Vec::new();
            let mut bind_group_entries = Vec::new();
            $(
                let buffer_binding_type = match $type {
                    Bbt::Uniform => BufferBindingType::Uniform,
                    Bbt::Storage{read_only} => BufferBindingType::Storage { read_only: read_only }
                };

                layout_entries.push(BindGroupLayoutEntry {
                    binding: $binding,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: buffer_binding_type,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });

                bind_group_entries.push(BindGroupEntry {
                    binding: $binding,
                    resource: $buffer.as_entire_binding(),
                });
            )*

            let bind_group_layout = $device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("{} Layout", $label)),
                entries: &layout_entries,
            });

            let bind_group = $device.create_bind_group(&BindGroupDescriptor {
                label: Some($label),
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            });

            (bind_group_layout, bind_group)
        }
    };
}
